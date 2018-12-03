import pandas as pd
import numpy as np

from utils import overlap, ptb_to_english


class Batcher:
    def __init__(self, data, is_train=True, target_batch_size=None):
        apid_order, apids, node_sizes, node_sizes_by_apid = [], [], [], {}

        for example in data:
            context_squad_tree, apid = example['context_squad_tree'], example['apid']

            # apid is a uuid for context paragraphs
            # It stands for "Article-Paragraph Id" and is the concatenation of article title and paragraph number.
            # I.e. "Beyonce:2 is the second paragraph of the Wikipedia article about Beyonce)
            apids.append(apid)

            # Paragraph contexts not repeated (occur just the first time)
            # So dataset will have multiple QA pairs over same passage
            # To save space and computation efforts, the paragraph is not repeated
            # The context modelling happens just once and is cloned for every QA pair which references it
            if context_squad_tree is None:
                assert apid == apid_order[-1]
            else:
                apid_order.append(apid)
                node_sizes_by_apid[apid] = context_squad_tree.num_nodes()
            node_sizes.append(node_sizes_by_apid[apid])

        order_df = pd.DataFrame({'apid': apids, 'node_size': node_sizes})
        order_df['order'] = [n for n in range(order_df.shape[0])]

        # Sort data by number of syntax constituents in passage
        order_df.sort_values(by=['node_size', 'apid'], inplace=True)
        batch_idxs = list(order_df.order)

        self.batches, current_batch = [], []
        for example_idx in batch_idxs:
            example = data[example_idx]
            if len(current_batch) >= target_batch_size and not example['apid'] == current_batch[-1]['apid']:
                self.batches.append(Batch(current_batch, is_train))
                current_batch = []
                assert example['context_tokens'] is not None
            current_batch.append(example)

        if len(current_batch) >= 0:
            self.batches.append(Batch(current_batch, is_train))

        self.batch_idx = 0
        self.reset()

    def reset(self):
        self.batch_idx = 0
        np.random.shuffle(self.batches)

    def next(self):
        batch = self.batches[self.batch_idx]
        self.batch_idx += 1
        return batch

    def has_next(self):
        has_next = self.batch_idx < self.num_batches()
        if not has_next:
            self.reset()
        return has_next

    def num_batches(self):
        return len(self.batches)


class Batch:
    def __init__(self, data, is_train):
        self.is_train = is_train
        self.apids, self.apid_order, self.trees, self.questions, self.answers, self.has_answer,\
            self.qa_id = [], [], [], [], [], [], []
        for example in data:
            # Article - Paragraph Id
            # TreeParagraph Object from representation
            apid, tree, question, answers, is_impossible, qa_id = (example['apid'],
                                                                   example['context_squad_tree'],
                                                                   example['question_tokens'],
                                                                   example['answers'],
                                                                   example['is_impossible'],
                                                                   example['qa_id'])
            if tree is None:
                assert apid == self.apids[-1]
            else:
                self.trees.append(tree)
                self.apid_order.append(apid)

            self.apids.append(apid)
            self.questions.append(ptb_to_english(question))
            self.answers.append([ptb_to_english(a) for a in answers])
            self.has_answer.append(1 if is_impossible == 0 else 0)
            self.qa_id.append(qa_id)

        self.f1_cache = self._f1s()
        self.context_tokens_cache = self._context_tokens()
        self.best_tokens, self.sentence_leaf_spans = self._context_best_sentence_tokens()
        self.predicted_answers = None

    def batch_size(self):
        return len(self.apids)

    def tree_idx_to_batch_idx(self, tree_idx):
        return self.apids.index(self.apid_order[tree_idx])

    def batch_idx_to_tree_idx(self, batch_idx):
        return self.apid_order.index(self.apids[batch_idx])

    def max_question_size(self):
        return max([len(q) for q in self.questions])

    def max_node_size(self):
        return max([t.num_nodes() for t in self.trees])

    def question_tokens(self):
        return self.questions

    def context_tokens(self):
        return self.context_tokens_cache

    def _context_tokens(self):
        l = []
        for tree in self.trees:
            l.append(ptb_to_english(tree.tokens))
        return l

    def context_best_sentence_tokens(self):
        return self.best_tokens, self.sentence_leaf_spans

    def sentence_tokens_from_node_idxs(self, node_idxs):
        _, _, sentence_leaf_spans = self._get_sentence_idxs(node_idxs)
        best_tokens = []

        for batch_idx in range(self.batch_size()):
            s = sentence_leaf_spans[batch_idx]
            sentence_tokens = self.trees[self.batch_idx_to_tree_idx(batch_idx)].tokens[s[0]: s[1]]
            best_tokens.append(sentence_tokens)
        return best_tokens, sentence_leaf_spans

    def update_predictions(self, node_idxs):
        self.predicted_answers = []
        for batch_idx in range(self.batch_size()):
            tree = self.trees[self.batch_idx_to_tree_idx(batch_idx)]
            answer = tree.span(node_idxs[batch_idx])
            self.predicted_answers.append([answer])

    def _context_best_sentence_tokens(self):
        best_nodes = np.argmax(self.f1s(), axis=-1)
        _, _, sentence_leaf_spans = self._get_sentence_idxs(best_nodes)
        best_tokens = []

        for batch_idx in range(self.batch_size()):
            s = sentence_leaf_spans[batch_idx]
            sentence_tokens = self.trees[self.batch_idx_to_tree_idx(batch_idx)].tokens[s[0]: s[1]]
            best_tokens.append(sentence_tokens)
        return best_tokens, sentence_leaf_spans

    def f1s(self):
        return self.f1_cache

    def _f1s(self):
        f1s = np.zeros([self.batch_size(), self.max_node_size()])
        for (batch_idx, apid) in enumerate(self.apids):
            tree = self.trees[self.batch_idx_to_tree_idx(batch_idx)]
            f1s[batch_idx, :tree.num_nodes()] = [overlap(tree.span(i), self.answers[batch_idx])[-1]
                                                 for i in range(tree.num_nodes())]

        return f1s

    def precision_recall(self, predicted_idxs):
        exact_matches, recalls, precisions, f1s, allf1s = [], [], [], [], []
        for (batch_idx, predicted_idx) in enumerate(predicted_idxs):
            tree = self.trees[self.batch_idx_to_tree_idx(batch_idx)]
            span = tree.span(predicted_idx)
            em, recall, precision, f1 = overlap(span, self.answers[batch_idx])

            allf1s.append(f1)

            if self.has_answer[batch_idx]:
                exact_matches.append(em); recalls.append(recall); precisions.append(precision)
                f1s.append(f1)

        return exact_matches, recalls, precisions, f1s, allf1s

    def _get_sentence_idxs(self, node_idxs):
        sentence_idxs, sentence_node_spans, sentence_leaf_spans = [], [], []

        for batch_idx, node_idx in enumerate(node_idxs):
            tree_idx = self.batch_idx_to_tree_idx(batch_idx)
            tree = self.trees[tree_idx]
            node = tree.get_node(node_idx)
            sentence_idxs.append(node.sentence_idx)
            sentence_node_spans.append(tree.sentence_node_ranges[node.sentence_idx])
            sentence_leaf_spans.append(tree.sentence_leaf_ranges[node.sentence_idx])
        return sentence_idxs, sentence_node_spans, sentence_leaf_spans

