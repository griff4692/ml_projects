from __future__ import print_function
import torch
import import_spinn
import torch.optim as optim
import torch.nn.functional as F
from batcher import prepare_fake_batches
from classifier import FakeClassifier
import numpy as np
import argparse
from constants import PAD, SHIFT, REDUCE
import sys
from utils import render_args, cudify
import json
import re
from prepro import get_meta

meta_by_sentence, meta_vocab = get_meta()

def add_num_ops(args, sent):
    trans = sent[1]
    max_ops = trans.size()[1]
    # find number of padding actions and subtract from max ops row-wise
    if args.gpu > -1:
        mask = trans.data.cpu().numpy().copy()
    else:
        mask = trans.data.numpy().copy()
    mask[mask > 0] = 0
    num_ops = max_ops + mask.sum(axis=1)
    return (sent[0], trans, num_ops)

def predict(args, model, sent1, meta, cuda=False):
    sent1 = add_num_ops(args, sent1)

    model.eval()
    output, _, _ = model(sent1, meta, None)
    logits = F.log_softmax(output)
    if args.gpu > -1:
        return logits.data.cpu().numpy().argmax(axis=1)
    return logits.data.numpy().argmax(axis=1)

def get_l2_loss(model, l2_lambda):
    loss = 0.0
    for w in model.parameters():
        if w.grad is not None:
            loss += l2_lambda * torch.sum(torch.pow(w, 2))
    return loss

def train_batch(args, model, loss, optimizer, sent1, y_val, meta, step, teacher_prob):
    sent1 = add_num_ops(args, sent1)

    # Reset gradient
    optimizer.zero_grad()
    # Forward
    fx, sent_true, sent_pred = model(sent1, meta, teacher_prob)
    logits = F.log_softmax(fx)

    total_loss = loss(logits, y_val)

    if args.teacher and sent_pred is not None and sent_true is not None:
        total_loss += args.teach_lambda * loss.forward(sent_pred, sent_true)

    total_loss += get_l2_loss(model, 1e-5)

    # Backward
    total_loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.clamp(-args.grad_clip, args.grad_clip)
    # Update parameters
    optimizer.lr = 0.001 * (0.75 ** (step / 10000.0))
    optimizer.step()
    return total_loss.data[0]

def remove_bad(batch, vocab):
    valid_s = []
    valid_t = []
    valid_labels = []
    valid_meta = []

    l = batch.label - 1

    sentences = batch.hypothesis.transpose(0, 1)
    transitions = batch.hypothesis_transitions.t() - 2
    batch_size = sentences.size()[0]
    for i in range(batch_size):
        if transitions[i, 0].data[0] == SHIFT and transitions[i, 1].data[0] == SHIFT:
            words = list(reversed([vocab.itos[idx] for idx in sentences[i].data]))
            words_no_pad = [w for w in words if not w in ['-rrb-', '-lrb-', "<pad>"]]
            words_str = "".join(words_no_pad).strip().lower()
            words_str = re.sub('[^A-Za-z0-9]+', '', words_str)

            if words_str in meta_by_sentence:
                meta = meta_by_sentence[words_str]
                valid_meta.append(meta)
            else:
                # print("Couldn't find\t%s" % words_str)
                continue

            valid_s.append(sentences[i].unsqueeze(0))
            valid_t.append(transitions[i].unsqueeze(0))
            valid_labels.append(l[i])

    return (torch.cat(valid_s), torch.cat(valid_t)), torch.cat(valid_labels), valid_meta

def train(args):
    print ("\nStarting...")
    sys.stdout.flush()
    label_names, (train_iter, dev_iter, test_iter, inputs) = prepare_fake_batches(args)
    num_labels = len(label_names)
    num_labels -= 1
    label_names = label_names[1:]

    print(label_names)

    print("Prepared Dataset...\n")

    sys.stdout.flush()
    model = FakeClassifier(args, inputs.vocab, meta_vocab)
    model.set_weight(inputs.vocab.vectors.numpy())

    print("Instantiated Model...\n")

    sys.stdout.flush()
    model = cudify(args, model)
    loss = torch.nn.NLLLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    count_iter = 0
    train_iter.repeat = False

    step = 0
    teacher_prob = 1.0

    for epoch in range(args.epochs):
        epoch_interp = float(args.epochs - epoch) / float(args.epochs)
        args.teach_lambda = (epoch_interp * args.teach_lambda_init) + ((1.0 - epoch_interp) * args.teach_lambda_end)
        train_iter.init_epoch()
        cost = 0
        for batch_idx, batch in enumerate(train_iter):
            model.train()
            step += 1
            count_iter += batch.batch_size

            input, label, meta = remove_bad(batch, inputs.vocab)
            cost += train_batch(args,
                model, loss, optimizer,
                input,
                label,
                meta,
                step,
                teacher_prob
            )

            if count_iter >= args.eval_freq:
                correct, total = 0.0, 0.0
                count_iter = 0
                confusion_matrix = np.zeros([num_labels, num_labels])
                dev_iter.init_epoch()

                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    model.eval()

                    input, label, meta = remove_bad(dev_batch, inputs.vocab)
                    pred = predict(
                        args,
                        model,
                        input,
                        meta
                    )
                    if args.gpu > -1:
                        true_labels = label.data.cpu().numpy()
                    else:
                        true_labels =  label.data.numpy()
                    for i in range(num_labels):
                        true_labels_by_cat = np.where(true_labels == i)[0]
                        pred_values_by_cat = pred[true_labels_by_cat]
                        num_labels_by_cat = len(true_labels_by_cat)
                        mass_so_far = 0
                        for j in range(num_labels - 1):
                            mass = len(pred_values_by_cat[pred_values_by_cat == j])
                            confusion_matrix[i, j] += mass
                            mass_so_far += mass

                        confusion_matrix[i, num_labels - 1] += num_labels_by_cat - mass_so_far

                    total += dev_batch.batch_size
                correct = np.trace(confusion_matrix)
                print ("Accuracy for batch #%d, epoch #%d --> %.1f%%\n" % (batch_idx, epoch, float(correct) / total * 100))
                true_label_counts = confusion_matrix.sum(axis=1)
                pred_label_counts = confusion_matrix.sum(axis=0).tolist()
                pred_label_counts = [str(int(c)) for c in pred_label_counts] + ["--> guessed distribution"]
                print("\nConfusion matrix (x-axis is true labels)\n")
                label_names = [n[0:8] + '.' for n in label_names]
                print("\t" + "\t".join(label_names) + "\n")
                for i in range(num_labels):
                    print (label_names[i], end="")
                    for j in range(num_labels):
                        if true_label_counts[i] == 0:
                            perc = 0.0
                        else:
                            perc = confusion_matrix[i, j] / true_label_counts[i]
                        print("\t%.2f%%" % (perc * 100), end="")
                    print("\t(%d examples)\n" % true_label_counts[i])

                print("\t" + "\t".join(pred_label_counts))
                print("")
                sys.stdout.flush()

        teacher_prob *= args.force_decay
        print("Cost for Epoch #%d --> %.2f\n" % (epoch, cost))
        torch.save(model, '../weights/model_%d.pth' % epoch)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='SPINN dependency parse + FAKE NEWS arguments.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--meta_embed_dim', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--grad_clip', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate to pass to optimizer.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('-continuous_stack', action='store_true', default=False)
    parser.add_argument('--eval_freq', type=int, default=50000, help='number of examples between evaluation on dev set.')
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--dropout_rate_input', type=float, default=0.1)
    parser.add_argument('--dropout_rate_classify', type=float, default=0.1)
    parser.add_argument('--classifier_dim', type=int, default=100)
    parser.add_argument('-no_batch_norm', action='store_true', default=False)
    parser.add_argument('-tracking', action='store_true', default=False)
    parser.add_argument('-teacher', action='store_true', default=False)
    parser.add_argument('--force_decay', type=float, default=1.0)
    parser.add_argument('--gpu', type=int, default=-1, help='-1 for cpu. 0 for gpu')
    parser.add_argument('--teach_lambda_init', type=float, default=4.0, help='relative contribution of FAKENEWS classifier versus dependency transitions to loss.')
    parser.add_argument('--teach_lambda_end', type=float, default=0.5)

    args = parser.parse_args()

    if args.debug:
        args.eval_freq = 1000

    if args.continuous_stack:
        assert args.tracking

    render_args(args)
    sys.stdout.flush()
    train(args)
