import json
import os

import argparse
import torch
from torch.nn import Sigmoid

from batcher import Batcher
from utils import load_pk, overlap, variable_to_numpy, tokens_to_text

from dominate.tags import *

NA_THRESHOLD = 0.28836128799999994
HAS_ANSWER_THRESHOLD = 1.0 - NA_THRESHOLD


def main(args):
    # Prepare output directory under ./weights/ to store model-specific data including weights
    weights_dir = 'weights/%s' % args.experiment

    # Load weights for best performing models on dev set and generate qualitative error report
    if args.cuda:
        span_extractor = torch.load(os.path.join(weights_dir, 'best_span_extractor.tar'))
        answer_verifier = torch.load(os.path.join(weights_dir, 'best_answer_verifier.tar'))
    else:
        span_extractor = torch.load(os.path.join(weights_dir, 'best_span_extractor.tar'), map_location='cpu')
        answer_verifier = torch.load(os.path.join(weights_dir, 'best_answer_verifier.tar'), map_location='cpu')
        span_extractor.use_cuda = False
        answer_verifier.use_cuda = False

    # Load Dev Data and save it to this model's weights dir
    print('Loading Dev Data...')
    dev_full_data = load_pk('preprocess/data/squad_dev_trees_evaluate_v2.0.npy')
    dev_batcher = Batcher(dev_full_data, is_train=False, target_batch_size=100)
    print('Done Loading Dev Data...')

    if args.cuda:
        span_extractor.cuda(); answer_verifier.cuda()

    span_f1 = 0.0
    span_em = 0.0
    overall_f1 = 0.0
    overall_em = 0.0
    verifier_accuracy = 0.0

    N = 0.0

    official_eval = {}
    tuning_data = []
    qa_map = json.load(open('preprocess/data/qa_id_map.json'))

    _html = html()
    _body = _html.add(body())
    _body.add(h1('Errors & Successes'))

    _body.add(h3('Predict has answer threshold = %.3f' % HAS_ANSWER_THRESHOLD))

    num_batches = dev_batcher.num_batches()
    batch_no = 0

    span_extractor.eval(); answer_verifier.eval()
    while dev_batcher.has_next():
        batch = dev_batcher.next()
        node_scores, expected_f1s, global_answer_score = span_extractor(batch, eval_system=True)
        score_confidence, predicted_node_idxs = node_scores.max(dim=1)
        score_confidence, predicted_node_idxs = (variable_to_numpy(score_confidence, args.cuda),
                                                 variable_to_numpy(predicted_node_idxs, args.cuda))

        # Answer score = predicted has answer probability
        answer_score = answer_verifier(batch, predicted_node_idxs=predicted_node_idxs, eval_system=True)
        answer_proba = variable_to_numpy(Sigmoid()(answer_score), args.cuda)  # convert from tensor to numpy array
        global_answer_proba = variable_to_numpy(Sigmoid()(global_answer_score), args.cuda)

        has_answer_proba = 0.3 * score_confidence + 0.4 * global_answer_proba + 0.3 * answer_proba

        predicted_spans = [
            batch.trees[batch.batch_idx_to_tree_idx(batch_idx)].span(predicted_node_idxs[batch_idx])
            for batch_idx in range(batch.batch_size())
        ]

        N += batch.batch_size()

        for batch_idx in range(batch.batch_size()):
            has_answer = batch.has_answer[batch_idx]
            predicted_has_answer = has_answer_proba[batch_idx] >= HAS_ANSWER_THRESHOLD

            batch_span_em, _, _, batch_span_f1 = overlap(predicted_spans[batch_idx], batch.answers[batch_idx])

            confidence = score_confidence[batch_idx]
            qa_id = batch.qa_id[batch_idx]

            predicted_text = tokens_to_text(predicted_spans[batch_idx], qa_map['passages'][qa_map['qa_id'][qa_id]])
            official_eval[qa_id] = predicted_text if predicted_has_answer else ''

            tree_idx = batch.batch_idx_to_tree_idx(batch_idx)
            supposed_batch_idx = batch.tree_idx_to_batch_idx(tree_idx)
            if batch_idx == supposed_batch_idx:
                _body.add(h1(batch.apids[batch_idx]))
                _body.add(p(qa_map['passages'][qa_map['qa_id'][qa_id]]))
            _body.add(h3(' '.join(batch.questions[batch_idx])))

            answer_strs = ', '.join(['\'' + ' '.join(a) + '\'' for a in batch.answers[batch_idx]])
            predicted_answer_str = 'Yes' if predicted_has_answer else 'No'
            _body.add(ul(
                div('True Answers: ' + answer_strs),
                div('Predicted Answer: \'' + predicted_text +
                    '\' (Confidence' + '= ' + str(round(confidence, 1)) + ')'),
                div('Span Metrics: F1 = %.2f, EM = %.2f.' % (batch_span_f1, batch_span_em)),
                div('True Has Answer: ' + ('Yes' if has_answer else 'No')),
                div('Predicted Has Answer: ' + predicted_answer_str + ' (' +
                    str(round(has_answer_proba[batch_idx], 1)) + ' > ' + str(round(HAS_ANSWER_THRESHOLD, 1)) + ')')
            ))

            span_f1 += batch_span_f1
            span_em += batch_span_em

            tuning_data.append({
                'qa_id': qa_id,
                'prediction': predicted_text,
                'prediction_span': ' '.join(predicted_spans[batch_idx]),
                'span_f1': str(span_f1),
                'true_has_answer': has_answer,
                'answer_confidence': str(confidence),
                'global_answer_proba': str(global_answer_proba[batch_idx]),
                'local_answer_proba': str(answer_proba[batch_idx])
            })

            if predicted_has_answer and has_answer:
                overall_f1 += batch_span_f1
                overall_em += batch_span_em
                verifier_accuracy += 1.0

            if not predicted_has_answer and not has_answer:
                overall_f1 += 1.0
                overall_em += 1.0
                verifier_accuracy += 1.0

        batch_no += 1
        if batch_no % 5 == 0:
            print('%d out of %d batches complete.' % (batch_no, num_batches))

            if batch_no == 5:
                with open(os.path.join(weights_dir, 'error_report.html'), 'w') as fd:
                    fd.write(_html.render())

    print('System Span F1: %.4f' % (span_f1 / N))
    print('System Span EM: %.4f' % (span_em / N))
    print('System Answer Verifier Accuracy: %.4f' % (verifier_accuracy / N))
    print('System Overall F1: %.4f' % (overall_f1 / N))
    print('System Overall EM: %.4f' % (overall_em / N))

    json.dump(official_eval, open(os.path.join(weights_dir, 'official_eval.json'), 'w'))
    json.dump(tuning_data, open(os.path.join(weights_dir, 'tuning_data.json'), 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Script for SQUAD Tree LSTM.')
    parser.add_argument('--experiment', default='evaluate-11-10-v3')
    parser.add_argument('-cuda', action='store_true', default=False, help="Whether or not to use cuda.")
    args = parser.parse_args()

    main(args)
