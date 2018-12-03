import json
import argparse
import os
import subprocess

import numpy as np


def generate_weights():
    c = []
    for i in np.arange(0.0, 1.1, 0.1):
        for j in np.arange(0.0, round(1.1 - i, 1), 0.1):
            c.append((abs(round(i, 1)), abs(round(j, 1)), abs(round(1.0 - i - j, 1))))

    return c


ORDER = ['span', 'global', 'local']


def generate_probs(args):
    path_to_tune_data = 'weights/%s/tuning_data.json' % args.experiment

    output_dir = 'weights/%s/permutations/' % args.experiment

    tuning_data = json.load(open(path_to_tune_data, 'r'))
    weights = generate_weights()
    W = len(weights)
    na_probs = []
    for _ in range(W):
        na_probs.append({})

    span_data = {}

    for dict in tuning_data:
        qa_id, prediction = dict['qa_id'], dict['prediction']
        span_data[qa_id] = prediction

        score_prob = float(dict['answer_confidence'])
        global_prob = float(dict['global_answer_proba'])
        local_prob = float(dict['local_answer_proba'])

        for na_prob_idx in range(len(weights)):
            coeff = weights[na_prob_idx]
            has_answer_prob = (score_prob * coeff[0]) + (global_prob * coeff[1]) + (local_prob * coeff[2])
            na_prob = 1.0 - has_answer_prob
            assert na_prob >= 0.0 and na_prob <= 1.0
            na_probs[na_prob_idx][qa_id] = na_prob

    for na_prob_idx in range(len(weights)):
        weight = weights[na_prob_idx]
        na_prob_list = na_probs[na_prob_idx]
        file_name = ORDER[0] + '_' + str(weight[0]) + '_' + ORDER[1] + '_' + str(weight[1]) + '_' + ORDER[2] + '_' + str(weight[2])
        file_path = output_dir + file_name + '.json'
        json.dump(na_prob_list, open(file_path, 'w'))

    json.dump(span_data, open(output_dir + 'official_span_eval.json', 'w'))


def run_eval_script(args):
    permutations_dir = 'weights/%s/permutations/' % args.experiment
    data_file = 'data/squad/dev-v2.0.json'
    weights = generate_weights()

    span_predictions = os.path.join(permutations_dir, 'official_span_eval.json')

    for na_prob_idx in range(len(weights)):
        weight = weights[na_prob_idx]
        na_prob_file = os.path.join(
            permutations_dir,
            ORDER[0] + '_' + str(weight[0]) + '_' + ORDER[1] + '_' + str(weight[1]) + '_' + ORDER[2] + '_' + str(weight[2]))
        out_file = na_prob_file + '_results.json'

        subprocess.call(['python', 'official_evaluation_script.py', data_file, span_predictions, '--na-prob-file',
                         na_prob_file + '.json', '--out-file', out_file])

        print(out_file, '->', json.load(open(out_file, 'r'))['best_f1'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune evaluation NA Prob values from output of evaluation script.')
    parser.add_argument('--experiment', default='evaluate-11-10-v3')
    parser.add_argument('-run_official_eval', action='store_true', default=False)
    args = parser.parse_args()

    args.run_official_eval = True

    if args.run_official_eval:
        run_eval_script(args)
    else:
        generate_probs(args)

# TODO remove
# import json
# 
# NA_THRESHOLD = 0.28836128799999994
# HAS_ANSWER_THRESHOLD = 1.0 - NA_THRESHOLD
# 
# path_to_tune_data = 'weights/evaluate-11-10-v3/tuning_data.json'
# tuning_data = json.load(open(path_to_tune_data))
# 
# output_dir = 'weights/evaluate-11-10-v3/tuned_official_eval.json'
# 
# span_data = {}
# 
# for dict in tuning_data:
#     qa_id, prediction = dict['qa_id'], dict['prediction']
# 
#     score_prob = float(dict['answer_confidence'])
#     global_prob = float(dict['global_answer_proba'])
#     local_prob = float(dict['local_answer_proba'])
# 
#     has_answer_proba = 0.3 * score_prob + 0.4 * global_prob + 0.3 * local_prob
# 
#     predicted_has_answer = has_answer_proba >= HAS_ANSWER_THRESHOLD
# 
#     span_data[qa_id] = prediction if predicted_has_answer else ''
# 
# json.dump(span_data, open(output_dir, 'w'))
