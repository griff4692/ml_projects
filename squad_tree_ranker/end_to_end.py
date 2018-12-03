import json
import os
import sys

import nltk
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.parse.stanford import StanfordParser
from nltk.tokenize import sent_tokenize
import torch
from torch.nn import Sigmoid

from batcher import Batch
from squad_tree import TreePassage
from utils import variable_to_numpy, tokens_to_text

dir_path = os.path.dirname(os.path.realpath(__file__))

class_path = (os.path.join(dir_path, 'stanford-postagger-2018-02-27/stanford-postagger.jar') + ':' +
              os.path.join(dir_path, 'stanford-parser-full-2018-02-27/stanford-parser.jar'))
os.environ['CLASSPATH'] = class_path
os.environ['STANFORD_MODELS'] = os.path.join(
    dir_path, 'stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar')

NA_THRESHOLD = 0.28836128799999994
HAS_ANSWER_THRESHOLD = 1.0 - NA_THRESHOLD

EXPERIMENT = os.path.join(dir_path, 'weights/evaluate-11-10-v3')


def main(in_path, outpath):
    nltk.download()

    span_extractor = torch.load(os.path.join(EXPERIMENT, 'best_span_extractor.tar'), map_location='cpu')
    answer_verifier = torch.load(os.path.join(EXPERIMENT, 'best_answer_verifier.tar'), map_location='cpu')
    span_extractor.use_cuda = False
    answer_verifier.use_cuda = False

    tokenizer = StanfordTokenizer(options={'ptb3Escaping': True})  # same tokenizer used by lexical parser
    parser = StanfordParser(java_options='-mx5g')

    data = json.load(open(in_path, 'r'))['data']
    batches = []
    official_eval = {}
    official_eval_tokens = {}
    qaid_map = {}

    num_articles = len(data)
    for aidx in range(len(data)):
        article = data[aidx]
        print('\t- Article Count=%d/%d' % (aidx + 1, num_articles))
        for pidx, paragraph in enumerate(article['paragraphs']):
            passage, qas = paragraph['context'], paragraph['qas']
            passage = passage.replace(u'\xa0', ' ')
            sentences = sent_tokenize(passage)

            sentence_tokens = [tokenizer.tokenize(sentence) for sentence in sentences]
            raw_trees = [list(s)[0] for s in list(parser.parse_sents(sentence_tokens, verbose=True))]
            squad_tree = TreePassage(raw_trees)

            for qidx, qa in enumerate(qas):
                question_sentences = sent_tokenize(qa['question'])
                question_tokens = []
                for s in question_sentences:
                    question_tokens += tokenizer.tokenize(s)

                batches.append(Batch([{
                    'apid': 'apid',
                    'qa_id': qa['id'],
                    'context_squad_tree': squad_tree,
                    'question_tokens': question_tokens,
                    'answers': [],
                    'is_impossible': 0
                }], False))

                qaid_map[qa['id']] = paragraph['context']

    span_extractor.eval(); answer_verifier.eval()
    for idx, batch in enumerate(batches):
        qa_id = batch.qa_id[0]

        node_scores, expected_f1s, global_answer_score = span_extractor(batch, eval_system=True)
        score_confidence, predicted_node_idxs = node_scores.max(dim=1)
        score_confidence, predicted_node_idxs = (variable_to_numpy(score_confidence, False),
                                                 variable_to_numpy(predicted_node_idxs, False))

        # Answer score = predicted has answer probability
        answer_score = answer_verifier(batch, predicted_node_idxs=predicted_node_idxs, eval_system=True)
        answer_proba = variable_to_numpy(Sigmoid()(answer_score), False)  # convert from tensor to numpy array
        global_answer_proba = variable_to_numpy(Sigmoid()(global_answer_score), False)

        has_answer_proba = (0.3 * score_confidence + 0.4 * global_answer_proba + 0.3 * answer_proba)[0]

        predicted_span = batch.trees[0].span(predicted_node_idxs[0])
        predicted_has_answer = has_answer_proba >= HAS_ANSWER_THRESHOLD

        predicted_text = tokens_to_text(predicted_span, qaid_map[qa_id])
        official_eval[qa_id] = predicted_text if predicted_has_answer else ''
        official_eval_tokens[qa_id] = ' '.join(predicted_span) if predicted_has_answer else ''

    json.dump(official_eval, open(outpath, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
