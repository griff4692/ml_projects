from collections import defaultdict
import re
from string import punctuation
import sys

import numpy as np
from six.moves import cPickle

from unicodedata import normalize


def load_pk(filename):
    fin = open(filename,"rb")
    object = cPickle.load(fin)
    fin.close()
    return object


def save_as_pk(data, filename):
    fout = open(filename,'wb')
    cPickle.dump(data, fout, protocol=cPickle.HIGHEST_PROTOCOL)
    fout.close()


def split_on_soft_dash(arr):
    new_arr = []
    for a in arr:
        new_arr += a.split(u'\xa0')
    return new_arr


def f1(a, b):
    if a + b == 0.0:
        return 0.0
    else:
        return 2 * (a * b) / (a + b)


_PTB_PUNCTUATION = ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
_PTB_TRANSLATION = ['(', ')', '{', '}', '[', ']']


def ptb_to_english(tokens):
    eng = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in _PTB_PUNCTUATION:
            eng.append(_PTB_TRANSLATION[_PTB_PUNCTUATION.index(token_lower)])
        else:
            eng.append(token)
    return eng


def is_punctuation(token):
    assert token == token.lower()
    if token in _PTB_PUNCTUATION:
        return True
    for char in token:
        if char not in punctuation:
            return False
    return True


_ARTICLES = ['a', 'an', 'the']


def is_article(token):
    return token in _ARTICLES


def remove_filler(tokens):
    filtered = []
    for token in tokens:
        token_lower = token.lower()
        if not is_punctuation(token_lower) and not is_article(token_lower):
            filtered.append(token_lower)

    return filtered


def special_match(prediction, reference):
    if not len(reference) == 1:
        return 0.0, 0.0

    overlaps = 0.0
    for token in prediction:
        if np.any(np.array(re.split("[" + punctuation + "]+", token)) == reference[0].lower()):
            overlaps = 1.0

    return overlaps / float(len(prediction)), min(overlaps, 1.0)


def overlap(prediction, references):
    exact_match = 0.0
    recalls, precisions = [], []

    prediction = remove_filler(prediction)
    pred_len = float(len(prediction))

    if pred_len == 0:
        return 0.0, 0.0, 0.0, 0.0

    if len(references) == 0.0:
        return 1.0, 1.0, 1.0, 1.0

    for reference in references:
        reference_counts = defaultdict(int)
        reference = remove_filler(reference)
        ref_len = max(float(len(reference)), 1)

        for token in reference:
            assert token == token.lower()
            reference_counts[token] += 1

        overlaps = 0.0
        for token in prediction:
            assert token == token.lower()
            reference_counts[token] -= 1
            if reference_counts[token] >= 0:
                overlaps += 1.0

        if reference == prediction:
            exact_match = 1.0

        precision, recall = overlaps / pred_len, overlaps / ref_len
        sp, sr = special_match(prediction, reference)
        precisions.append(max(precision, sp)); recalls.append(max(recall, sr))
        assert recall <= 1.0 and precision <= 1.0
    f1s = np.array([f1(precision, recall) for precision, recall in zip(precisions, recalls)])
    best_idx = int(np.argmax(f1s))
    return exact_match, recalls[best_idx], precisions[best_idx], f1s[best_idx]


def contains_answer(token, answers):
    tokens = remove_filler([token])

    if len(tokens) == 0:
        return False

    token = tokens[0]

    for answer in answers:
        answer = remove_filler(answer)
        if token in answer:
            return True

    return False


def print_and_log(string, logger):
    print(string)
    sys.stdout.flush()
    logger.write(string + "\n")
    logger.flush()


def format_seconds(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    if m == 0:
        return '%d seconds' % s
    if h == 0:
        return '%d minutes, %d seconds' % (m, s)
    else:
        return "%d hours, %d minutes" % (h, m)


def variable_to_numpy(v, use_cuda=True):
    return v.data.cpu().numpy() if use_cuda else v.data.numpy()


def truncate(string, tokens_to_match, truncate_from_start=True):
    N = len(string)

    if N == 0:
        return string

    N = len(string)

    if len(tokens_to_match) > 1:
        tokens_to_match.append(tokens_to_match[0] + tokens_to_match[1])
        tokens_to_match.append(tokens_to_match[-1] + tokens_to_match[-2])

    if truncate_from_start:
        for i in range(N):
                truncated_first = string[i:]
                if truncated_first in tokens_to_match:
                    return truncated_first
        return string
    else:
        for i in range(N):
            truncated_last = string[:N-i]
            if truncated_last in tokens_to_match:
                return truncated_last

    # Might be double compound
    biggest_size = 0
    for i in range(N):
        for j in range(i + 1, N + 1):
            snippet = string[i:j]
            if snippet in tokens_to_match and len(snippet) > biggest_size:
                biggest_size = len(snippet)
                truncated_last = snippet
    if truncated_last is None:
        raise Exception('Uh Oh!')
    return truncated_last


def tokens_to_text(tokens, text):
    """
    :param tokens: Array of tokenized data i.e. ['obama', 'was', 'better']
    :param text: Text in which to locate tokens i.e. 'Obama was better than Trump.'
    :return: Span of text from which tokens were extracted 'Obama was better'
    """
    tokens = ptb_to_english(tokens)

    tokens_truncated = re.sub("[\W]+", '', ''.join(tokens).lower())
    text_split = text.split(' ')

    for start_idx in range(len(text_split)):
        for end_idx in range(start_idx + 1, len(text_split) + 1):
            snippet = text_split[start_idx:end_idx]
            snippet_truncated = re.sub("[\W]+", '', ''.join(snippet).lower())

            if snippet_truncated == tokens_truncated:
                return ' '.join(snippet).strip('&*,-./:;=?@^_\'"`')

    # Answer must be part of punctuated expression
    smallest_substring = 100000
    best_snippet = None
    for start_idx in range(len(text_split)):
        for end_idx in range(start_idx + 1, len(text_split) + 1):
            snippet = text_split[start_idx:end_idx]
            snippet_truncated = re.sub("[\W]+", '', ''.join(snippet).lower())

            if tokens_truncated in snippet_truncated:
                if len(snippet_truncated) < smallest_substring:
                    best_snippet = snippet
                    smallest_substring = len(snippet_truncated)

    if best_snippet is None:
        return ' '.join(tokens)

    # Strip tailing punctuated expressions
    best_snippet[0] = truncate(best_snippet[0], tokens, truncate_from_start=True)
    best_snippet[-1] = truncate(best_snippet[-1], tokens, truncate_from_start=False)
    best_snippet_truncated = re.sub("[\W]+", '', ''.join(best_snippet).lower())

    if not tokens_truncated == best_snippet_truncated:
        print('\n\n\n\nLook out below')
        print(best_snippet, tokens)
        print('\n\n\n\n')
    return ' '.join(best_snippet).strip('&*,-./:;=?@^_\'"`')


if __name__ == '__main__':
    import json
    qa_id_map = json.load(open('preprocess/data/qa_id_map.json', encoding='utf-8'))

    data = load_pk('preprocess/data/squad_dev_trees_v2.0.npy')

    for example in data:
        qa_id = example['qa_id']
        answer = example['answers'][0]

        text_idx = qa_id_map[qa_id]
        text = qa_id_map['passages'][text_idx]
        proper_answer = tokens_to_text(answer, text)
        print(answer, ' -> ',  proper_answer)
