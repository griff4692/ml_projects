import argparse
import numpy as np

from preprocess.generate_squad_tokens import split_unk_compound_token
from utils import load_pk, save_as_pk


def generate_embed_matrix(vocab, embed_dim):
    glove_path = 'data/glove.6B/glove.6B.%dd.txt' % embed_dim
    glove_dict, glove_vals = {}, []

    with open(glove_path, 'r', encoding='utf-8') as fd:
        for line in fd:
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if len(word) == 0 or type(word) == float:
                print('Invalid GloVe word %s' % word)
                continue
            glove_vals += vector
            glove_dict[word] = vector

    matrix = np.array(glove_vals).std() * np.random.randn(vocab.size(), embed_dim)
    unk_tokens = []
    for i in range(vocab.size()):
        token = vocab.get_token(i)
        assert token == token.lower()
        if token in glove_dict:
            matrix[i] = glove_dict[token]
        else:
            compound = split_unk_compound_token(token, glove_dict)
            if len(compound) > 0:
                for (sub_idx, sub_token) in enumerate(compound):
                    sub_token_embeds = np.array(glove_dict[sub_token])
                    compound_embeds = sub_token_embeds if sub_idx == 0 else np.add(compound_embeds, sub_token_embeds)
                matrix[i] = compound_embeds
            else:
                unk_tokens.append(token)

    # print('%d tokens randomly initialized because not in GloVe: %s' % (len(unk_tokens), ', '.join(unk_tokens)))
    return matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Embedding Matrix for training and dev data.')
    parser.add_argument('-mini', action='store_true', default=False, help='To generate mini version of dataset.')
    args = parser.parse_args()

    mini_str = '/mini' if args.mini else ''
    vocab = load_pk('data%s/vocab.npy' % mini_str)

    embedding_matrix = generate_embed_matrix(vocab)

    embedding_out_path = 'data%s/embedding_matrix.npy' % mini_str
    save_as_pk(embedding_matrix, embedding_out_path)
    print('Saved embedding matrix to %s' % embedding_out_path)
