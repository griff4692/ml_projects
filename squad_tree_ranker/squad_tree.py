from nltk.tree import Tree


"""
    Creates Flattened Pre-Order representations of arrays of nltk.Tree objects
    Where each nltk.Tree object represents the syntax parse tree for a single sentence
    Tree's leaves represent the original sentence words
    
    TreePassage is simply concatenation of TreeSentences and represents a single SQuAD context
"""


class TreePassage:
    def __init__(self, passage_trees):
        self.tokens, self.nodes, self.sentence_node_ranges, self.sentence_leaf_ranges = [], [], [], []
        sentence_start_node, sentence_start_leaf_idx = 0, 0
        for sentence_idx, tree in enumerate(passage_trees):
            tree_sentence = TreeSentence(sentence_idx, tree, sentence_start_idx=sentence_start_leaf_idx)
            self.nodes += tree_sentence.nodes
            self.tokens += tree.leaves()
            num_nodes = tree_sentence.num_nodes()

            self.sentence_node_ranges.append((sentence_start_node, sentence_start_node + num_nodes))
            self.sentence_leaf_ranges.append((
                sentence_start_leaf_idx, sentence_start_leaf_idx + tree_sentence.num_leaves))
            sentence_start_node += num_nodes
            sentence_start_leaf_idx += tree_sentence.num_leaves

        self._check_children()

    def _check_children(self):
        for parent_idx in range(self.num_nodes()):
            self.get_child_idxs(parent_idx)  # will raise Exception if children don't fully encompass parent's span

    def num_nodes(self):
        return len(self.nodes)

    def num_leaves(self):
        return len(self.tokens)

    def get_node(self, idx):
        return self.nodes[idx]

    def leaf_order(self):
        return [node_idx for (node_idx, node) in enumerate(self.nodes) if node.is_leaf]

    def get_child_idxs(self, parent_idx):
        child_ids = self.nodes[parent_idx].child_ids
        child_idxs = [node_idx for node_idx, node in enumerate(self.nodes) if node.id in child_ids]
        assert len(child_idxs) == len(child_ids)
        return child_idxs

    def span(self, node_idx):
        span_start, span_end = self.nodes[node_idx].span
        return self.tokens[span_start: span_end + 1]

    def render(self, f1s=None):
        for i in range(len(self.nodes)):
            span_str = ' '.join(self.span(i))
            print('%d: %s' % (i + 1, span_str))
            if f1s is not None and f1s[i] > 0.0:
                print('\t\tF1=%.2f' % f1s[i])
            child_ids = self.get_child_idxs(i)
            if len(child_ids) > 0:
                child_strs = []
                for child in child_ids:
                    child_strs.append(' '.join(self.span(child)))
                print('\t\t%s' % ' ~  '.join(child_strs))


class TreeSentence:
    def __init__(self, sentence_idx, tree, sentence_start_idx=0):
        leaves = tree.leaves()
        self.num_leaves = len(leaves)
        tree.collapse_unary(collapsePOS = True, collapseRoot = True)
        preorder_positions = tree.treepositions(order='preorder')

        self.nodes, span_start = [], 0
        for position in preorder_positions:
            subtree = tree[position]
            if isinstance(subtree, Tree):
                subtree_span = TreeSentence.claim_spans(subtree.leaves(), leaves, span_start)
                subtree_span = (subtree_span[0] + sentence_start_idx, subtree_span[1] + sentence_start_idx)
                self.nodes.append(Node(subtree, sentence_idx, subtree_span))
            else:
                span_start += 1

    def num_nodes(self):
        return len(self.nodes)

    def num_leaves(self):
        return self.num_leaves

    def leaf_order(self):
        return [node_idx for (node_idx, node) in enumerate(self.nodes) if node.is_leaf]

    @classmethod
    def claim_spans(cls, subtree_leaves, sentence_leaves, span_start_idx):
        span_length = len(subtree_leaves)
        for start_idx in range(span_start_idx, len(sentence_leaves) - span_length + 1):
            candidate_span_leaves = sentence_leaves[start_idx:start_idx + span_length]
            if subtree_leaves == candidate_span_leaves:
                candidate_span = (start_idx, start_idx + span_length - 1)
                return candidate_span
        raise Exception('Could not find child span.')


class Node:
    """
        Node's contain their span information in terms of index ranges
        as well as unique identifiers in the form of "Sentence Number: Span Start: Span End"
        These UUIDs allow parents to find their children in the flattened version of the tree
    """

    def __init__(self, tree, sentence_idx, span):
        self.label = tree.label()  # POS TAG TODO Could leverage this in representation
        self.height = tree.height()  # Height from node to word-level leaf node
        self.sentence_idx = sentence_idx  # In which sentence does this node span
        self.span = span  # start and end index of leaf word-level span
        self.id = '%d:%d-%d' % (self.sentence_idx, self.span[0], self.span[1])
        self.is_leaf = isinstance(tree[0], str)
        self.child_ids = []
        if self.is_leaf:
            assert self.height == 2
        else:  # Find what children's UUIDs (id) will be for quick retrieval later
            child_start_span = self.span[0]
            child_lens = [len(child.leaves()) for child in tree]
            child_spans= []
            for (child_idx, child_len) in enumerate(child_lens):
                start_idx = child_start_span if child_idx == 0 else child_spans[-1][1] + 1
                end_idx = start_idx + child_len - 1
                child_spans.append((start_idx, end_idx))

            assert child_spans[0][0] == self.span[0] and child_spans[-1][1] == self.span[1]
            self.child_ids = ['%d:%d-%d' % (self.sentence_idx, span[0], span[1]) for span in child_spans]
