import numpy as np
import json

from utils import load_pk, overlap

data = load_pk('./data/squad_dev_trees_v2.0.npy') + load_pk('./data/squad_train_trees_v2.0.npy')
question_ngrams = {}
pos_counts = {}
unigram_counts = {}
trigram_counts = {}

unique_trigrams = set()

q_words = ['who', 'why', 'when', 'how', 'which', 'whose', 'where', 'whom', 'what']

for idx, d in enumerate(data):
    tree = d['context_squad_tree']
    answers = d['answers']
    j = idx - 1
    while tree is None:
        tree = data[j]['context_squad_tree']
        j -= 1

    f1s = np.array([overlap(tree.span(i), answers)[-1] for i in range(tree.num_nodes())])
    best_idx = np.argmax(f1s)
    best_node = tree.nodes[best_idx]
    best_pos = best_node.label

    if best_pos not in pos_counts:
        pos_counts[best_pos] = 1
    else:
        pos_counts[best_pos] += 1

    q = d['question_tokens']
    q_tree = d['question_squad_tree']
    question_pos = [q_tree.nodes[idx].label for idx in q_tree.leaf_order()]

    for qi in range(len(q)):
        t = q[qi].lower()

        if t in q_words:
            ngrams = [t]

            if t not in unigram_counts:
                unigram_counts[t] = 1
            else:
                unigram_counts[t] += 1

            prev_pos = question_pos[qi - 1] if qi > 0 else '<BOS>'
            next_pos = question_pos[qi + 1] if qi + 1 < len(q) else '<EOS>'
            trigram = prev_pos + '_' + t + '_' + next_pos
            ngrams.append(trigram)

            unique_trigrams.add(trigram)

            if trigram not in trigram_counts:
                trigram_counts[trigram] = 1
            else:
                trigram_counts[trigram] += 1

            for ngram in ngrams:
                if ngram not in question_ngrams:
                    question_ngrams[ngram] = {}

                if best_pos not in question_ngrams[ngram]:
                    question_ngrams[ngram][best_pos] = 1
                else:
                    question_ngrams[ngram][best_pos] += 1

            break  # only look at first question word for now

print(len(data), len(unique_trigrams))

json.dump(question_ngrams, open('/Users/griffinadams/Desktop/NeuralQA_advancement/griffin-squad/preprocess/data/question_ngram_counts.json', 'w'))
json.dump(unigram_counts, open('/Users/griffinadams/Desktop/NeuralQA_advancement/griffin-squad/preprocess/data/unigram_counts.json', 'w'))
json.dump(trigram_counts, open('/Users/griffinadams/Desktop/NeuralQA_advancement/griffin-squad/preprocess/data/trigram_counts.json', 'w'))
json.dump(pos_counts, open('/Users/griffinadams/Desktop/NeuralQA_advancement/griffin-squad/preprocess/data/pos_counts.json', 'w'))
