import os

import argparse
import numpy as np
import torch

from calculate_pos_prob import calculate_pos_prob
from utils import load_pk, overlap, variable_to_numpy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Script for SQUAD Tree LSTM.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('-cuda', action='store_true', default=False)
    args = parser.parse_args()

    # Prepare output directory under ./weights/ to store model-specific data including weights
    out_dir = 'weights/%s' % args.experiment

    # Load Dev Data and save it to this model's weights dir
    dev_data = load_pk('preprocess/data/squad_dev_trees_v2.0.npy')
    dev_batcher = load_pk(os.path.join(out_dir, 'dev_batcher.npy'))

    # Load vocab and generate embedding matrix
    vocab = load_pk(os.path.join(out_dir, 'vocab.npy'))

    # Load best performing models on dev set
    span_extractor = torch.load(os.path.join(out_dir, 'best_span_extractor.tar'), map_location='cpu')
    answer_verifier = torch.load(os.path.join(out_dir, 'best_answer_verifier.tar'), map_location='cpu')

    span_extractor.use_cuda = False
    answer_verifier.use_cuda = False

    if args.cuda:
        span_extractor.cuda()
        answer_verifier.cuda()

    span_extractor.eval()
    answer_verifier.eval()

    old_metrics, new_metrics = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    num_changed = 0.

    while dev_batcher.has_next():
        batch = dev_batcher.next()
        node_scores, expected_f1s = span_extractor(batch, vocab)

        span_confidence, predicted_span_idxs = node_scores.max(dim=1)
        span_confidence, predicted_span_idxs, node_scores = (variable_to_numpy(span_confidence, args.cuda),
                                                             variable_to_numpy(predicted_span_idxs, args.cuda),
                                                             variable_to_numpy(node_scores, args.cuda))

        batch_size = batch.batch_size()
        for batch_idx in range(batch.batch_size()):
            tree_idx = batch.batch_idx_to_tree_idx(batch_idx)
            tree = batch.trees[tree_idx]
            question = batch.questions[batch_idx]
            answers = batch.answers[batch_idx]
            apid = batch.apids[batch_idx]

            best_idx = predicted_span_idxs[batch_idx]
            predicted_pos = tree.nodes[best_idx].label
            tree_pos = [n.label for n in tree.nodes]

            span_prob = node_scores[batch_idx, :]

            q_tree = batch.q_trees[batch_idx]
            question_pos = [q_tree.nodes[idx].label for idx in q_tree.leaf_order()]

            pos_scores = [calculate_pos_prob(n.label, question, question_pos) for n in tree.nodes]
            pos_prob = np.exp(pos_scores) / np.sum(np.exp(pos_scores))

            old_span = tree.nodes[best_idx].span
            candidate_idxs = []
            for i in range(len(tree.nodes)):
                n = tree.nodes[i]
                span = n.span

                # a0, b0, a1, b2
                if span[0] <= old_span[0] and span[1] >= old_span[0]:
                    candidate_idxs.append(i)
                elif old_span[0] <= span[0] and old_span[1] >= span[0]:
                    candidate_idxs.append(i)
                else:
                    continue

            candidate_span_prob, candidate_pos_prob = span_prob[candidate_idxs], pos_prob[candidate_idxs]
            candidate_joint_prob = [candidate_span_prob[j] * candidate_pos_prob[j] for j in range(len(candidate_idxs))]
            reranked_best_idx = candidate_idxs[np.argmax(np.array(candidate_joint_prob))]
            candidate_tree_pos = np.array(tree_pos)[candidate_idxs]

            if not best_idx == reranked_best_idx:
                num_changed += 1
                old_pos, new_pos = predicted_pos, tree.nodes[reranked_best_idx].label
                old_span, new_span = tree.span(best_idx), tree.span(reranked_best_idx)

                _, old_recall, old_precision, old_f1 = overlap(old_span, batch.answers[batch_idx])
                _, new_recall, new_precision, new_f1 = overlap(new_span, batch.answers[batch_idx])

                old_metrics = np.add(old_metrics, np.array([old_f1, old_recall, old_precision]))
                new_metrics = np.add(new_metrics, np.array([new_f1, new_recall, new_precision]))

                if new_f1 == old_f1:
                    print('No material change')
                elif new_f1 > old_f1:
                    print('Improvement from %.3f to %.3f' % (old_f1, new_f1))
                else:
                    print('Worsened from %.3f to %.3f' % (old_f1, new_f1))
                print('\t' + ' '.join(question))
                print('\t' + 'Old span: %s (%s)' % (' '.join(old_span), old_pos))
                print('\t' + 'New span: %s (%s)' % (' '.join(new_span), new_pos))

    n = float(num_changed)
    old_metrics /= n
    new_metrics /= n
    print('Old metrics: F1=%.4f, Recall=%.4f, Precision=%.4f' % (old_metrics[0], old_metrics[1], old_metrics[2]))
    print('New metrics: F1=%.4f, Recall=%.4f, Precision=%.4f' % (new_metrics[0], new_metrics[1], new_metrics[2]))
