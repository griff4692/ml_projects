from datetime import datetime
from math import exp
import os
import shutil
from time import time

import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, Sigmoid

from batcher import Batcher
from evaluator import Evaluator
from tree_lstm import TreeLSTM
from answer_verifier import AnswerVerifier
from utils import format_seconds, load_pk, print_and_log, save_as_pk, variable_to_numpy


def _run_batch(batch=None, span_extractor=None, span_extractor_optimizer=None, answer_verifier=None,
               answer_verifier_optimizer=None, answer_verifier_logistic_loss=None, evaluator=None):
    # Clear gradients and get next batch
    span_extractor_optimizer.zero_grad(); answer_verifier_optimizer.zero_grad()

    # First dimension is batch size for each of the following variables
    # Node Scores = Predicted softmax Probabilities for each constituent (Probability it's span is answer)
    # Expected F1s = E(F1) Per example, it is the dot product of Node Scores and per-node F1 scores
    node_scores, expected_f1s, global_answer_score = span_extractor(batch)
    span_confidence, predicted_node_idxs = node_scores.max(dim=1)
    span_confidence, predicted_node_idxs = (variable_to_numpy(span_confidence, args.cuda),
                                            variable_to_numpy(predicted_node_idxs, args.cuda))
    batch_span_loss = (-expected_f1s).mean()

    # Answer score = predicted has answer probability
    answer_score = answer_verifier(batch, predicted_node_idxs=predicted_node_idxs)
    # Needs to be a Variable to be passable to PyTorch's loss functions
    has_answer_tensor = span_extractor.cudify(Variable(torch.FloatTensor(batch.has_answer), requires_grad=False))
    answer_verifier_loss = answer_verifier_logistic_loss(answer_score, has_answer_tensor)

    global_answer_verifier_loss = answer_verifier_logistic_loss(global_answer_score, has_answer_tensor)

    # Calculate joint loss and back propagate
    joint_loss = batch_span_loss + 2.0 * answer_verifier_loss + 2.0 * global_answer_verifier_loss

    answer_proba = variable_to_numpy(Sigmoid()(answer_score), args.cuda)  # convert from tensor to numpy array
    global_answer_proba = variable_to_numpy(Sigmoid()(global_answer_score), args.cuda)
    evaluator.add_predictions(batch=batch, span_loss=1 - exp(-batch_span_loss.item()),
                              answer_loss=answer_verifier_loss.item(), predicted_node_idxs=predicted_node_idxs,
                              answer_proba=answer_proba, span_confidence=span_confidence,
                              global_answer_loss=global_answer_verifier_loss.item(),
                              global_answer_proba=global_answer_proba, has_answer=batch.has_answer)

    return joint_loss


def main(args):
    mini_str = '/mini' if args.mini else ''  # path to mini dataset
    version_suffix = '_v2.0' if args.squad_version == 2.0 else ''  # gets proper dataset version (1.1 or 2.0)

    # Prepare output directory under ./weights/ to store model-specific data including weights
    out_dir = 'weights/%s' % args.experiment
    if os.path.exists(out_dir):
        print('Warning - you are overwriting previous experiment %s. Hit Ctrl Z to abort.\n' % args.experiment)
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    logger = open(os.path.join(out_dir, 'log.txt'), 'w')
    print_and_log('Timestamp = %s for %s\n' %
                  (datetime.strftime(datetime.now(), '%m/%d/%Y %H:%M'), args.experiment), logger)

    # Load Dev Data and save it to this model's weights dir
    print_and_log('Loading v%s Dev Data...' % args.squad_version, logger)
    dev_data = load_pk('preprocess/data%s/squad_dev_trees%s.npy' % (mini_str, version_suffix))
    dev_batcher = Batcher(dev_data, is_train=False, target_batch_size=args.batch_size)
    save_as_pk(dev_batcher, os.path.join(out_dir, 'dev_batcher.npy'))
    print_and_log('Loaded Dev Data...', logger)

    # Load Train Data and save it to this model's weights dir
    print_and_log('Loading v%s Train Data...' % args.squad_version, logger)
    train_data = load_pk('preprocess/data%s/squad_train_trees%s.npy' % (mini_str, version_suffix))
    train_batcher = Batcher(train_data, is_train=True, target_batch_size=args.batch_size)
    print_and_log('Loaded Train Data...', logger)

    # Create models and optimizers
    span_extractor = TreeLSTM(use_cuda=args.cuda)
    answer_verifier = AnswerVerifier(use_cuda=args.cuda)

    if args.cuda:
        span_extractor.cuda(); answer_verifier.cuda()

    span_extractor_grad_params = filter(lambda p: p.requires_grad, span_extractor.parameters())
    span_extractor_optimizer = optim.Adam(span_extractor_grad_params, args.span_extractor_lr)

    answer_verifier_grad_params = filter(lambda p: p.requires_grad, answer_verifier.parameters())
    answer_verifier_optimizer = optim.Adam(answer_verifier_grad_params, args.answer_verifier_lr)

    # Determines if question is answerable or not
    answer_verifier_logistic_loss = BCEWithLogitsLoss(pos_weight=span_extractor.cudify(torch.FloatTensor([0.5])))

    best_span_f1 = -1  # Keep track of which epoch model achieves highest span level F1 on the dev set
    best_answer_verifier_accuracy = -1
    best_span_epoch = -1
    best_answer_verifier_epoch = -1
    for epoch_idx in range(args.epochs):
        print_and_log('Starting Epoch %d...' % (epoch_idx + 1), logger)

        train_evaluator = Evaluator('train')  # Stores predictions and returns evaluation string at the end of epoch
        dev_evaluator = Evaluator('dev')

        start_time = time()

        span_extractor.train(); answer_verifier.train()
        while train_batcher.has_next():
            # Clear gradients and get next batch
            span_extractor_optimizer.zero_grad(); answer_verifier_optimizer.zero_grad()

            joint_loss = _run_batch(batch=train_batcher.next(), span_extractor=span_extractor,
                                    span_extractor_optimizer=span_extractor_optimizer, answer_verifier=answer_verifier,
                                    answer_verifier_optimizer=answer_verifier_optimizer,
                                    answer_verifier_logistic_loss=answer_verifier_logistic_loss,
                                    evaluator=train_evaluator)

            joint_loss.backward()

            # Make a gradient step
            span_extractor_optimizer.step(); answer_verifier_optimizer.step()
        print_and_log('Took %s.' % format_seconds(time() - start_time), logger)
        print_and_log('\t' + train_evaluator.eval_string(), logger)

        span_extractor.eval(); answer_verifier.eval()
        while dev_batcher.has_next():
            _run_batch(batch=dev_batcher.next(), span_extractor=span_extractor,
                       span_extractor_optimizer=span_extractor_optimizer, answer_verifier=answer_verifier,
                       answer_verifier_optimizer=answer_verifier_optimizer,
                       answer_verifier_logistic_loss=answer_verifier_logistic_loss, evaluator=dev_evaluator)

        print_and_log('\t' + dev_evaluator.eval_string(), logger)
        dev_f1 = dev_evaluator.span_f1()
        if dev_f1 > best_span_f1:
            best_span_f1 = dev_f1
            best_span_epoch = epoch_idx + 1
            torch.save(span_extractor, os.path.join(out_dir, 'best_span_extractor.tar'))

        dev_answer_verifier_accuracy = dev_evaluator.avg_answer_accuracy()
        if dev_answer_verifier_accuracy > best_answer_verifier_accuracy:
            best_answer_verifier_accuracy = dev_answer_verifier_accuracy
            best_answer_verifier_epoch = epoch_idx + 1
            torch.save(answer_verifier, os.path.join(out_dir, 'best_answer_verifier.tar'))

    print_and_log('\nBest span = %.4f F1 at %d epoch' % (best_span_f1, best_span_epoch), logger)
    print_and_log('\nBest answer verifier = %.4f accuracy at %d epoch' % (
        best_answer_verifier_accuracy, best_answer_verifier_epoch), logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Script for SQUAD Tree LSTM.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--span_extractor_lr', type=float, default=0.001, help="Learning rate for Adam")
    parser.add_argument('--answer_verifier_lr', type=float, default=0.0008, help="Learning rate for Adam")
    parser.add_argument('--squad_version', type=float, default=2.0)
    parser.add_argument('-cuda', action='store_true', default=False, help="Whether or not to use cuda.")
    parser.add_argument('-mini', action='store_true', default=False, help='To run on mini version of dataset.')
    args = parser.parse_args()

    main(args)
