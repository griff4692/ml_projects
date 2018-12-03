from collections import defaultdict
import json
import os

import argparse
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tokenize import sent_tokenize
import pandas as pd

from preprocess.constants import CONSTANTS
from preprocess.preprocess_utils import split_unk_compound_token

os.environ['CLASSPATH'] = '../stanford-postagger-2018-02-27/stanford-postagger.jar'


def add_tokens(token_counts, token_lists, glove_tokens):
    for token_list in token_lists:
        for token in token_list:
            token_lower = token.lower()
            token_counts[token_lower] += 1
            if token_lower not in glove_tokens:
                for token_lower_split_unk in split_unk_compound_token(token_lower, glove_tokens):
                    token_counts[token_lower_split_unk] += 1


def generate_squad_tokens(args, tokenizer, glove_tokens):
    mini_str = 'mini/' if args.mini else ''
    token_counts = defaultdict(int)
    train_data, dev_data = [], []
    unique_apids = set()

    types = ['train'] if args.skip_dev else ['dev'] if args.dev_only else ['dev', 'train']

    for type in types:
        data = json.load(open('../data/squad/%s-v%s.json' % (type, CONSTANTS['SQUAD_VERSION']), 'rb'))
        data = data['data']
        data_to_store = train_data if type == 'train' else dev_data
        num_articles = len(data)
        num_total_examples = 0
        print('%s: Generating tokenized data for %d articles.' %
              (type.capitalize(), num_articles))

        args.end_idx = min(args.end_idx, len(data))
        for aidx in range(args.start_idx, args.end_idx):
            article = data[aidx]

            article_title = article['title']
            print('\t- Article Count=%d/%d' % (aidx + 1, num_articles))

            num_paragraphs = len(article['paragraphs'])
            for pidx, paragraph in enumerate(article['paragraphs']):
                if (pidx + 1) % 25 == 0 or (pidx + 1) == num_paragraphs:
                    print('\t\t- Paragraph Count=%d/%d' % (pidx + 1, num_paragraphs))

                if pidx == 2 and args.mini:
                    break

                apid = '%s:%d' % (article_title, pidx)
                passage, qas = paragraph['context'], paragraph['qas']
                passage = passage.replace(u'\xa0', ' ')
                num_total_examples += len(qas)
                sentences = sent_tokenize(passage)

                sentence_tokens = [tokenizer.tokenize(sentence) for sentence in sentences]
                add_tokens(token_counts, sentence_tokens, glove_tokens)

                if len(sentence_tokens) > CONSTANTS['MAX_CONTEXT_SENTENCES'] and type == 'train':
                    print('Dropping example. Paragraph has %d sentences.  Can have at most %d.' %
                          (len(sentence_tokens), CONSTANTS['MAX_CONTEXT_SENTENCES']))
                    continue

                sentence_tokens_lengths = [len(s) for s in sentence_tokens]
                max_context_sentence_length, passage_token_length = (
                    max(sentence_tokens_lengths), sum(sentence_tokens_lengths))
                if max_context_sentence_length > CONSTANTS['MAX_CONTEXT_SENTENCE_LENGTH'] and type == 'train':
                    print('Dropping example. Paragraph has a sentence of length %d.  Max sentence length is %d.' %
                          (max_context_sentence_length,  CONSTANTS['MAX_CONTEXT_SENTENCE_LENGTH']))
                    continue

                if passage_token_length < CONSTANTS['MIN_SEQUENCE_LENGTH'] and type == 'train':
                    print('Dropping example. Paragraph is too short.  Has %d tokens.  Min tokens is %d.' %
                          (passage_token_length, CONSTANTS['MIN_SEQUENCE_LENGTH']))
                    continue

                for qidx, qa in enumerate(qas):
                    question_sentences = sent_tokenize(qa['question'])
                    if len(question_sentences) > CONSTANTS['MAX_QUESTION_SENTENCES'] and type == 'train':
                        print('Dropping example. Question has %d sentences.  Can have at most %d.' %
                              (len(question_sentences), CONSTANTS['MAX_QUESTION_SENTENCES']))
                        continue
                    question_tokens = tokenizer.tokenize(question_sentences[0])

                    question_token_length = len(question_tokens)
                    if question_token_length < CONSTANTS['MIN_SEQUENCE_LENGTH'] and type == 'train':
                        print('Dropping example. Question is too short.  Has %d tokens.  Min tokens is %d.' %
                              (question_token_length, CONSTANTS['MIN_SEQUENCE_LENGTH']))
                        continue

                    is_impossible = 0 if not CONSTANTS['SQUAD_VERSION'] == 2.0 else 1 if qa['is_impossible'] else 0
                    if is_impossible:
                        assert len(qa['answers']) == 0
                        if len(qa['plausible_answers']) == 0:
                            print('No plausible answers provided for impossible question.')

                    add_tokens(token_counts, [question_tokens], glove_tokens)
                    answers = qa['answers'] if is_impossible == 0 else qa['plausible_answers']
                    answer_counts = {}
                    for answer in answers:
                        start = answer['answer_start']
                        answer_key = answer['text'] + '::' + str(start)
                        if answer_key in answer_counts:
                            v = answer_counts[answer_key]
                            new_v = (v[0], v[1], v[2] + 1)
                        else:
                            tokenized_answer = tokenizer.tokenize(answer['text'])
                            new_v = (tokenized_answer, start, 1)
                        answer_counts[answer_key] = new_v

                    if apid in unique_apids:
                        sentence_tokens = None
                    unique_apids.add(apid)

                    data_to_store.append({
                        'article_title': article_title,
                        'paragraph_idx': pidx,
                        'apid': apid,
                        'qa_id': qa['id'],
                        'context_tokens': sentence_tokens,
                        'question_tokens': question_tokens,
                        'answers': list(answer_counts.values()),
                        'is_impossible': is_impossible
                    })

        version_suffix = '_v2.0' if CONSTANTS['SQUAD_VERSION'] == 2.0 else ''
        range_str = '_%d_%d' % (args.start_idx, args.end_idx)
        store_out_path = 'data/%ssquad_%s_tokens%s%s.json' % (mini_str, type, version_suffix, range_str)
        json.dump(data_to_store, open(store_out_path, 'w'))
        print('Saved %d tokenized examples out of %d for %s set to %s.' % (
            len(data_to_store), num_total_examples, type, store_out_path))

    vocab_df = pd.DataFrame([list(item) for item in token_counts.items()], columns=['token', 'count'])
    return vocab_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenization Pre-processing script for training and dev data.')
    parser.add_argument('-mini', action='store_true', default=False, help='To generate mini version of dataset.')
    parser.add_argument('-skip_dev', action='store_true', default=False, help='To skip dev.')
    parser.add_argument('-dev_only', action='store_true', default=False, help='To only do dev.')
    parser.add_argument('--start_idx', default=0, type=int)
    parser.add_argument('--count', default=50, type=int)
    args = parser.parse_args()

    # how many examples to process (if done in parallel to speed up)
    if args.mini:
        args.count = 2
    args.end_idx = args.start_idx + args.count

    tokenizer = StanfordTokenizer(options={'ptb3Escaping': True})  # same tokenizer used by lexical parser

    glove_csv = pd.read_csv('data/glove_tokens.csv')
    glove_tokens = set(glove_csv[glove_csv.token.notnull()].token)

    vocab_df = generate_squad_tokens(args, tokenizer, glove_tokens)

    if not args.dev_only and not args.skip_dev:
        range_str = '_%d_%d' % (args.start_idx, args.end_idx)
        mini_str = 'mini/' if args.mini else ''
        version_suffix = '_v2.0' if CONSTANTS['SQUAD_VERSION'] == 2.0 else ''
        vocab_out_path = 'data/%ssquad_vocab%s%s.csv' % (mini_str, version_suffix, range_str)
        vocab_df.to_csv(vocab_out_path, index=False)
        print('Saved vocabulary counts to %s' % vocab_out_path)
        print('Done tokenization for train and dev set!')
