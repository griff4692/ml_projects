import pandas as pd

import preprocess.constants as constants


def generate_glove_vocab(glove_in_path):
    glove_tokens = set()
    with open(glove_in_path, 'r', encoding='utf-8') as fd:
        for line in fd:
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            if len(word) == 0 or type(word) == float:
                print('Invalid GloVe word %s' % word)
                continue
            glove_tokens.add(word)
            assert word.lower() == word

    glove_df = pd.DataFrame(list(glove_tokens), columns=['token'])
    glove_df.sort_values('token', inplace=True)

    return glove_df


if __name__ == '__main__':
    glove_in_path = '../data/glove.6B/glove.6B.%sd.txt' % constants.EMBED_DIM
    glove_df = generate_glove_vocab(glove_in_path)
    print('Token Count in GloVe is %d.' % glove_df.shape[0])
    glove_out_path = 'data/glove_tokens.csv'
    glove_df.to_csv(glove_out_path, index=False)
    print('Saved GloVe tokens to %s.' % glove_out_path)
