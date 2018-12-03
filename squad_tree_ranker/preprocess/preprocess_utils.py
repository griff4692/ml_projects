import json, os, re

import pandas as pd


def split_unk_compound_token(token, glove_tokens):
    """
    :param token: compound token not found in GloVE
    :param glove_tokens: set of GloVe tokens
    :return: provide sub tokens which are recognized by GloVE.
    """
    assert token == token.lower()
    split_tokens = re.split('\W+', token)
    return [split_token for split_token in split_tokens if len(split_token) > 0 and split_token in glove_tokens]


def merge_file_chunks(file_path, file_patterns, interval, N, df=None):
    start_idx = 0
    interval_str = '%d_%d' % (start_idx, min(N, start_idx + interval))
    is_json = 'json' in file_patterns

    arr, df = [], None

    chunk_path = os.path.join(file_path, file_patterns % interval_str)
    while os.path.exists(chunk_path):
        print(chunk_path)
        if is_json:
            chunk = json.load(open(chunk_path, 'rb'))
            arr += chunk
        else:
            chunk_df = pd.read_csv(chunk_path)
            chunk_df.set_index(['token'])
            if df is None:
                df = chunk_df
            else:
                df = pd.merge(df, chunk_df, on='token', how='outer')
                df.fillna(0, inplace=True)
                df['count'] = df['count_x'] + df['count_y']
                df = df[['token', 'count']]

        start_idx += interval
        interval_str = '%d_%d' % (start_idx, min(N, start_idx + interval))
        chunk_path = os.path.join(file_path, file_patterns % interval_str)

    out_path = os.path.join(file_path, file_patterns % 'full')
    if is_json:
        json.dump(arr, open(out_path, 'w'))
    else:
        df.to_csv(out_path)


def remove_invalid_dev_questions(file_path):
    arr = json.load(open(file_path, 'rb'))
    valid_arr = []
    for a in arr:
        if len(a['answers']) > 0:
            valid_arr.append(a)
        else:
            assert a['context_tokens'] is None
    json.dump(valid_arr, open(file_path, 'w'))


if __name__ == '__main__':
    # merge_file_chunks('./data', 'squad_train_tokens_v2.0_%s.json', 50, 442)
    # remove_invalid_dev_questions('./data/squad_dev_tokens_v2.0.json')

    dev_vocab_df = pd.read_csv('./data-chunks/squad_vocab_v2.0_0_35.csv')
    dev_vocab_df.set_index(['token'])
    merge_file_chunks('./data-chunks', 'squad_vocab_v2.0_%s.csv', 50, 442, df=dev_vocab_df)
