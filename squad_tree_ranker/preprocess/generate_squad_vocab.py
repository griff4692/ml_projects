import argparse
import pandas as pd

from preprocess.constants import CONSTANTS
from preprocess.preprocess_utils import split_unk_compound_token
from preprocess.vocab import Vocab, UNK_TOKEN
from utils import save_as_pk


def generate_squad_ids(vocab, squad_vocab_counts_df, glove_tokens):
    glove_vocab_df = pd.DataFrame([[str(token), True] for token in list(glove_tokens)], columns=['token', 'is_glove'])
    merged_df = squad_vocab_counts_df.merge(glove_vocab_df, on='token', how='left')
    merged_df.is_glove = merged_df.is_glove.fillna(False)
    merged_df['should_include'] = merged_df.apply(
        lambda row: row['is_glove'] or row['count'] >= CONSTANTS['MIN_VOCAB_OCCURRENCE'], axis=1)

    possible_compound_tokens = list(set(merged_df[merged_df['should_include'] == False].token))
    compound_tokens = [t for t in possible_compound_tokens if len(split_unk_compound_token(t, list(glove_tokens))) > 0]

    tokens = list(set(merged_df[merged_df['should_include'] == True].token))

    full_tokens = [t for t in tokens + compound_tokens if type(t) == str and len(t) > 0]
    for (df_idx, token) in enumerate(full_tokens):
        actual_id = vocab.add(token)
        assert actual_id == df_idx + 2  # for unk and pad
        assert vocab.get_token(actual_id) == token
        assert vocab.get_id(token) == actual_id

    # assert <unk> works properly
    assert vocab.get_token(vocab.get_id('definitely-not-in-vocabulary')) == UNK_TOKEN

    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ID Vocab for training and dev data.')
    parser.add_argument('-mini', action='store_true', default=False, help='To generate mini version of dataset.')
    args = parser.parse_args()

    version_suffix = '_v2.0' if CONSTANTS['SQUAD_VERSION'] == 2.0 else ''
    mini_str = '/mini' if args.mini else ''

    glove_csv = pd.read_csv('data/glove_tokens.csv')
    glove_tokens = set(glove_csv[glove_csv.token.notnull()].token)

    squad_vocab_counts_df = pd.read_csv('data%s/squad_vocab%s.csv' % (mini_str, version_suffix))
    other_vocab_counts_df = pd.read_csv('data/squad_vocab.csv')

    vocab = Vocab()

    generate_squad_ids(vocab, squad_vocab_counts_df, glove_tokens)

    n, N = vocab.size(), squad_vocab_counts_df.shape[0]
    print('Saving %d out of %d possible words to vocabulary. %d unique tokens (or %.1f%%) deemed \'<unk>\'.' % (
        n, N, N - n, float(N - n)/float(N) * 100.0))

    vocab_out_path = 'data%s/vocab%s.npy' % (mini_str, version_suffix)
    save_as_pk(vocab, vocab_out_path)
    print('Saved vocabulary to %s' % vocab_out_path)
