import argparse

from preprocess.constants import CONSTANTS
from squad_tree import TreePassage, TreeSentence
from utils import load_pk, save_as_pk, split_on_soft_dash


def generate_squad_trees(raw_trees):
    full_tree_data, num_context_trees = [], 0

    for (example_idx, example) in enumerate(raw_trees):
        if example['context_raw_tree'] is None:
            example['context_squad_tree'] = None
        else:
            example['context_tokens'] = [split_on_soft_dash(s) for s in example['context_tokens']]
            example['context_squad_tree'] = TreePassage(example['context_raw_tree'])
            for sentence_idx, tree in enumerate(example['context_raw_tree']):
                assert tree.leaves() == example['context_tokens'][sentence_idx]

        example['answers'] = [split_on_soft_dash(answer[0]) for answer in example['answers']]
        example['question_squad_tree'] = TreeSentence(0, example['question_raw_tree'])
        full_tree_data.append(example)

        if (example_idx + 1) % 1000 == 0:
            print('%d out of %d examples processed...' % (example_idx + 1, len(raw_trees)))

    return full_tree_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tree Pre-processing script for training and dev data.')
    parser.add_argument('-mini', action='store_true', default=False, help='To generate mini version of dataset.')
    parser.add_argument('-dev_only', action='store_true', default=False)
    args = parser.parse_args()

    mini_str = '/mini' if args.mini else ''
    version_suffix = '_v2.0' if CONSTANTS['SQUAD_VERSION'] == 2.0 else ''

    categories = ['dev'] if args.dev_only else ['dev', 'train']

    for category in categories:
        print('Generating %s squad trees...' % category)
        raw_data_in_path = 'data%s/squad_%s_raw_trees%s.npy' % (mini_str, category, version_suffix)
        tree_data = generate_squad_trees(load_pk(raw_data_in_path))

        out_path = 'data%s/squad_%s_trees%s.npy' % (mini_str, category, version_suffix)
        save_as_pk(tree_data, out_path)
        print('Saved %s squad trees to %s' % (category, out_path))
