import json
import os

import argparse
from nltk.parse.stanford import StanfordParser

from preprocess.constants import CONSTANTS
from utils import save_as_pk


os.environ['CLASSPATH'] = '../stanford-parser-full-2018-02-27/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '../stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar'


def _generate_raw_trees(data, parser):
    tokens_to_parse, apid, qa_id = [], [], []

    for (example_idx, example) in enumerate(data):
        context_tokens, question_tokens = example['context_tokens'], example['question_tokens']
        if context_tokens is not None:
            for sentence in context_tokens:
                tokens_to_parse.append(sentence)
                apid.append(example['apid'])
                qa_id.append(None)

        tokens_to_parse.append(question_tokens)
        apid.append(None)
        qa_id.append(example['qa_id'])

    raw_trees = [list(s)[0] for s in list(parser.parse_sents(tokens_to_parse, verbose=True))]
    return list(zip(raw_trees, apid, qa_id))


def generate_raw_trees(tokenized_data, raw_tree_data):
    full_tree_data = []

    for (example_idx, example) in enumerate(tokenized_data):
        if example['context_tokens'] is None:
            example['context_raw_tree'] = None
        else:
            example['context_raw_tree'] = []
            num_context_sentences = len(example['context_tokens'])
            for _ in range(num_context_sentences):
                raw_tree, apid, qa_id = raw_tree_data.pop(0)
                assert apid == example['apid'] and qa_id is None
                example['context_raw_tree'].append(raw_tree)

        raw_tree, apid, qa_id = raw_tree_data.pop(0)
        assert apid is None and example['qa_id'] == qa_id
        example['question_raw_tree'] = raw_tree

        full_tree_data.append(example)

    assert len(raw_tree_data) == 0
    return full_tree_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tree Pre-processing script for training and dev data.')
    parser.add_argument('-mini', action='store_true', default=False, help='To generate mini version of dataset.')
    parser.add_argument('-dev_only', action='store_true', default=False)
    args = parser.parse_args()

    mini_str = '/mini' if args.mini else ''
    parser = StanfordParser(java_options='-mx5g')

    categories = ['dev'] if args.dev_only else ['dev', 'train']

    for category in categories:
        print('Generating %s squad trees...' % category)

        version_suffix = '_v2.0' if CONSTANTS['SQUAD_VERSION'] == 2.0 else ''
        tokenized_data_in_path = 'data%s/squad_%s_tokens%s.json' % (mini_str, category, version_suffix)
        tokenized_data = json.load(open(tokenized_data_in_path))

        tree_data = generate_raw_trees(tokenized_data, _generate_raw_trees(tokenized_data, parser))

        out_path = 'data%s/squad_%s_raw_trees%s.npy' % (mini_str, category, version_suffix)
        save_as_pk(tree_data, out_path)
        print('Saved %s squad raw trees to %s' % (category, out_path))
