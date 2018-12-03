from math import exp
import json

from utils import load_pk

q_words = ['who', 'why', 'when', 'how', 'which', 'whose', 'where', 'whom', 'what']

question_ngrams = json.load(open('./preprocess/data/question_ngram_counts.json', 'rb'))
pos_counts = json.load(open('./preprocess/data/pos_counts.json', 'rb'))


def calculate_pos_prob(pos, question_tokens, question_pos):
    unigram, trigram = None, None

    pos_V = len(pos_counts.keys())

    for qi in range(len(question_tokens)):
        q_token = question_tokens[qi].lower()

        if q_token in q_words:
            unigram = q_token
            prev_pos = question_pos[qi - 1] if qi > 0 else '<BOS>'
            next_pos = question_pos[qi + 1] if qi + 1 < len(question_tokens) else '<EOS>'
            trigram = prev_pos + '_' + unigram + '_' + next_pos

            break

    if trigram is None or unigram is None:  # Means we have 'other' question type
        return 1.0

    trigram_counts = question_ngrams[trigram]
    trigram_N = sum([trigram_counts[k] for k in trigram_counts])

    BASELINE = 1000.0
    trigram_adjustment = (trigram_N - BASELINE) / BASELINE
    trigram_lambda = 1.0 / (1.0 + exp(-trigram_adjustment))

    unigram_counts = question_ngrams[unigram]
    unigram_N = sum([unigram_counts[k] for k in unigram_counts])

    trigram_pos_ct = trigram_counts[pos] if pos in trigram_counts else 0.0
    trigram_prob = (trigram_pos_ct + 1.0) / float(trigram_N + pos_V)

    unigram_pos_ct = unigram_counts[pos] if pos in unigram_counts else 0.0
    unigram_prob = (unigram_pos_ct + 1.0) / float(unigram_N + pos_V)

    return trigram_lambda * trigram_prob + (1.0 - trigram_lambda) * unigram_prob


if __name__ == '__main__':
    dev_data = load_pk('/Users/griffinadams/Desktop/NeuralQA_advancement/griffin-squad/preprocess/data/squad_dev_trees_v2.0.npy')

    example = dev_data[0]
    question_pos = [example['question_squad_tree'].nodes[idx].label for idx in example['question_squad_tree'].leaf_order()]

    print(calculate_pos_prob('NP', example['question_tokens'], question_pos))
    print(calculate_pos_prob('S', example['question_tokens'], question_pos))
