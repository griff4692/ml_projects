import os

from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

options_file = os.path.join(os.getcwd(), 'data/elmo/options.json')
weight_file = os.path.join(os.getcwd(), 'data/elmo/weights.hdf5')

NEG_INF = -1e7


class TreeLSTM(nn.Module):
    def __init__(self, model_dim=100, use_cuda=False, dropout_p=0.5):
        super(TreeLSTM, self).__init__()
        self.use_cuda = use_cuda
        self.model_dim = model_dim

        # Set up embeddings as not trainable
        self.embeddings = Elmo(options_file, weight_file, 2, dropout=0, requires_grad=False)

        # Span extraction weights
        self.init_lstm_state = self.cudify(Variable(torch.zeros(1, 1, self.model_dim), requires_grad=False))
        self.encoder = nn.LSTM(
            input_size=2048, hidden_size=self.model_dim, batch_first=True, bidirectional=False, num_layers=1)

        self.modelling = nn.LSTM(input_size=self.model_dim * 8, hidden_size=self.model_dim * 2, batch_first=True,
                                 bidirectional=True, num_layers=2, dropout=dropout_p)

        # Tree LSTM weights for span extraction tree representation learning
        self.gates = nn.Linear(self.model_dim * 2, 6 * self.model_dim, bias=True)
        self.f_gate = nn.Linear(self.model_dim * 2, self.model_dim * 2, bias=True)
        self.output = nn.Linear(self.model_dim * 4, 1, bias=True)

        self.qsummary_projection = nn.Linear(self.model_dim * 4, self.model_dim * 4, bias=True)
        self.best_node_projection = nn.Linear(self.model_dim * 4, self.model_dim * 4, bias=True)
        self.global_answer_score = nn.Linear(self.model_dim * 16, 1, bias=True)

        # Shared functions / operators
        self.softmax= nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def _child_sum_lstm_ready(self, i, o, g, fc):
        c = torch.mul(i, g) + fc
        h = torch.mul(o, self.tanh(c))
        return h, c

    def _attn_mask(self, batch, row_sizes=None, col_sizes=None, is_self=False):
        if is_self:
            assert row_sizes == col_sizes

        # Batch_size, max node size, max question size
        max_r, max_c = max(row_sizes), max(col_sizes)
        attn_mask = self.cudify(torch.ByteTensor(batch.batch_size(), max_r, max_c))
        attn_mask.zero_()
        for batch_idx in range(batch.batch_size()):
            r, c = row_sizes[batch_idx], col_sizes[batch_idx]
            if r < max_r:
                attn_mask[batch_idx, r:, :] = 1
            if c < max_c:
                attn_mask[batch_idx, :, c:] = 1

            if is_self:
                for j in range(max_r):  # can't have attention with self
                    attn_mask[batch_idx, j, j] = 1
        return attn_mask

    def _bi_attn(self, c, q, attn_mask):
        # This is BiDAF: returns co-attended context
        assert c.size(0) == q.size(0)
        assert c.size(2) == q.size(2)

        attn = torch.bmm(c, q.transpose(1, 2))
        normalizer = np.power(c.size(2), 0.5)

        attn = attn / normalizer  # [batch_size, max_node, qs_len]
        attn.data.masked_fill_(attn_mask, NEG_INF)

        c2q_attn = self.softmax(attn)
        c2q = torch.bmm(c2q_attn, q)

        q2c_attn = self.softmax(attn.max(dim=-1, keepdim=False)[0]).unsqueeze(1)  # [batch_size, 1, max_node]
        q2c = torch.bmm(q2c_attn, c).repeat(1, c.size(1), 1)

        output = torch.cat([c, c2q, c * c2q, c * q2c], dim=-1)
        return output

    def cudify(self, tensor):
        if self.use_cuda:
            return tensor.cuda()
        return tensor

    def _encode_trees(self, batch, leaf_state):
        leaf_h, leaf_c = leaf_state

        states = self.cudify(Variable(torch.zeros(batch.batch_size(), batch.max_node_size(), 2, self.model_dim * 2)))

        todo, complete = set(), set()

        # Encode all leaf nodes with word-level values from the Bi-LSTM encoders
        for tree_idx, tree in enumerate(batch.trees):
            batch_idx = batch.tree_idx_to_batch_idx(tree_idx)
            apid = batch.apid_order[tree_idx]

            h = leaf_h[tree_idx, :tree.num_leaves(), :].unsqueeze(1)
            c = leaf_c[tree_idx, :tree.num_leaves(), :].unsqueeze(1)
            h_c = torch.cat([h, c], dim=1)

            leaf_idxs = tree.leaf_order()
            for (leaf_order, node_idx) in enumerate(leaf_idxs):
                states[batch_idx, node_idx, :, :] = h_c[leaf_order]          # the dimension looks weird ######

            processed = set([(apid, node_idx) for node_idx in leaf_idxs])
            all_nodes = set([(apid, node_idx) for node_idx in range(tree.num_nodes())])
            complete = complete.union(processed)
            todo = todo.union(all_nodes) - processed

        # Greedily batch process all (apid, node_idx) pairs in "to\do" as soon as all their children are in complete
        while len(todo) > 0:
            ready, offsets, ready_hs, ready_hs_sum, ready_cs = [], [0], [], [], []

            for (apid, node_idx) in todo:
                tree_idx = batch.apid_order.index(apid)
                tree = batch.trees[tree_idx]

                batch_idx = batch.tree_idx_to_batch_idx(tree_idx)
                child_idxs = tree.get_child_idxs(node_idx)
                children_plus_apid = [(apid, child_idx) for child_idx in child_idxs]

                # Process this node iff all it's children have already been processed
                if set(children_plus_apid).issubset(complete):
                    offsets.append(offsets[-1] + len(child_idxs))

                    child_idxs = self.cudify(Variable(torch.LongTensor(child_idxs)))
                    child_states = torch.index_select(states[batch_idx], 0, child_idxs)

                    child_hs, child_cs = child_states.split(1, dim=1)
                    child_hs_sum = child_hs.sum(dim=0)

                    ready.append((apid, node_idx))
                    ready_hs.append(child_hs);
                    ready_hs_sum.append(child_hs_sum);
                    ready_cs.append(child_cs)

            ready_hs = torch.cat(ready_hs, dim=0)

            # process all parents
            gates = self.gates(torch.cat(ready_hs_sum))
            (i, o, g) = gates.chunk(3, dim=1)
            i, o, g = self.sigmoid(i), self.sigmoid(o), self.tanh(g)

            f_gates = self.sigmoid(self.f_gate(ready_hs))
            cs = torch.cat(ready_cs)
            fc = torch.mul(f_gates, cs)  # total_child_num

            fc_reduce = []
            for idx in range(len(offsets) - 1):
                offset_s, offset_e = offsets[idx], offsets[idx + 1]
                fc_reduce.append(fc[offset_s:offset_e].sum(0))

            fc_reduce = torch.cat(fc_reduce)
            h, c = self._child_sum_lstm_ready(i, o, g, fc_reduce)
            h_c = torch.cat([h.unsqueeze(1), c.unsqueeze(1)], dim=1)

            for parent_ready_idx, (apid, node_idx) in enumerate(ready):
                tree_idx = batch.apid_order.index(apid)
                batch_idx = batch.tree_idx_to_batch_idx(tree_idx)
                states[batch_idx, node_idx, :, :] = h_c[parent_ready_idx]

            ready = set(ready)
            todo -= ready
            complete = complete.union(ready)

        # Clone passage state for all duplicated paragraphs
        # (i.e. QA pairs referencing a paragraph which has already been modeled)
        for batch_idx, apid in enumerate(batch.apids):
            first_tree_idx = batch.apids.index(apid)
            if not batch_idx == first_tree_idx:
                states[batch_idx, :, :, :] = states[first_tree_idx, :, :, :].clone()
        return states

    def _lstm_encoder(self, input):
        # Bi-LSTM implementation which returns hidden and cell states for every timestep
        timesteps = input.split(1, 1)      # assume this would split to (batch, 1, embed_size)  #####
        n = len(timesteps)

        for_hs, back_hs, for_cs, back_cs = [None] * n, [None] * n, [None] * n, [None] * n

        # Forward pass
        init_h, init_c = (self.init_lstm_state.repeat(1, input.size()[0], 1),
                          self.init_lstm_state.repeat(1, input.size()[0], 1))
        prev_h, prev_c = init_h, init_c
        for i in range(n):
            _, (step_h, step_c) = self.encoder(timesteps[i], (prev_h, prev_c))    # the input require to be (seq, batch, embed_size) need to check timesteps content #####
            for_hs[i] = step_h
            for_cs[i] = step_c
            prev_h, prev_c = step_h, step_c

        prev_h, prev_c = init_h, init_c
        for i in range(n - 1, -1, -1):
            _, (step_h, step_c) = self.encoder(timesteps[i], (prev_h, prev_c))
            back_hs[i] = step_h
            back_cs[i] = step_c
            prev_h, prev_c = step_h, step_c

        for_hs, for_cs = torch.cat(for_hs), torch.cat(for_cs)
        back_hs, back_cs = torch.cat(back_hs), torch.cat(back_cs)

        full_hs = torch.cat([for_hs, back_hs], dim=2).transpose(0, 1)
        full_cs = torch.cat([for_cs, back_cs], dim=2).transpose(0, 1)

        last_for_hs, last_back_hs = for_hs[-1], back_hs[0]
        summary_h = torch.cat([last_for_hs, last_back_hs], dim=1)

        last_for_cs, last_back_cs = for_cs[-1], back_cs[0]
        summary_c = torch.cat([last_for_cs, last_back_cs], dim=1)

        return (summary_h, summary_c), (full_hs, full_cs)

    def _output_mask(self, batch):
        output_mask = self.cudify(torch.ByteTensor(batch.batch_size(), batch.max_node_size()))
        output_mask.zero_()
        for batch_idx in range(batch.batch_size()):
            tree = batch.trees[batch.batch_idx_to_tree_idx(batch_idx)]
            tree_nodes = tree.num_nodes()
            if tree_nodes < batch.max_node_size():
                output_mask[batch_idx, tree_nodes:] = 1
        return output_mask

    def _feature_learning(self, word_matrix):
        """
            Passes ELMO embeddings through Bi-LSTM (returning hidden state for each timestep)
        """

        # Convert to tensors
        elmo = self.embeddings(self.cudify(word_matrix))
        word_embed = torch.cat(elmo['elmo_representations'], dim=2)

        # Pass Final state through Bi-LSTM
        (summary_h, summary_c), (h, c) = self._lstm_encoder(word_embed)

        return (summary_h, summary_c), (h, c)

    def forward(self, batch, eval_system=False):
        # self.encoder.flatten_parameters(); self.modelling.flatten_parameters()

        # Store batch sizes for padding masks (used later)
        c_node_sizes = [batch.trees[batch.batch_idx_to_tree_idx(i)].num_nodes() for i in range(batch.batch_size())]
        q_sizes = [len(batch.questions[i]) for i in range(batch.batch_size())]

        # Learn features for question words
        (qsummary_h, qsummary_c), (question_h, question_c) = self._feature_learning(batch_to_ids(batch.question_tokens()))
        question_states = question_h + question_c  # Empirically works better than just taking hidden states

        # Learn features for context syntax tree nodes/span
        # First by learning word level features
        _, (context_h, context_c) = self._feature_learning(batch_to_ids(batch.context_tokens()))

        # Then, encode Passage Tree Nodes bottom-up with Child-Sum Tree LSTM
        # Initialize leaf nodes with their word-level representations from previous step
        # Parents are LSTM-gated combination of children in syntax tree
        context_tree_states = self._encode_trees(batch, (context_h, context_c))
        context_tree_states = context_tree_states.sum(dim=2)  # Sum hidden and cell states for final representation

        # Model attention bi-directionally
        bidaf_mask = self._attn_mask(batch, row_sizes=c_node_sizes, col_sizes=q_sizes, is_self=False)
        modelling_input = self._bi_attn(context_tree_states, question_states, bidaf_mask)
        modelling_input = self.dropout(modelling_input)

        # Fuse BiDAF output via modelling layer
        classifier_input, _ = self.modelling(modelling_input)

        # Classify each tree node and mask padded node output with negative inf before taking softmax
        output = self.output(self.dropout(classifier_input)).squeeze(2)
        output.data.masked_fill_(self._output_mask(batch), NEG_INF)

        # Outputs probability for each syntax constituent representing whether or not it's span is the best answer
        out_soft = self.softmax(output)

        # Generate expected F1s as dot product of node scores and node true f1 scores
        batch_f1s = self.cudify(torch.FloatTensor(batch.f1s()))
        expected_val = torch.mul(out_soft, batch_f1s).sum(dim=1)

        if eval_system:
            _, best_node_idxs = out_soft.max(dim=-1)
            batch.update_predictions(best_node_idxs)
        else:
            # Get tensor of node states which maximize F1
            # To be used for classifying whether or not question is answerable
            best_node_idxs = np.argmax(batch.f1s(), axis=-1)

        best_nodes = self.cudify(Variable(torch.FloatTensor(batch.batch_size(), self.model_dim * 4)))
        for batch_idx in range(batch.batch_size()):
            best_nodes[batch_idx, :] = classifier_input[batch_idx, best_node_idxs[batch_idx], :]

        qsummary = self.relu(self.qsummary_projection(self.dropout(torch.cat([qsummary_h, qsummary_c], dim=-1))))
        best_nodes_proj = self.relu(self.best_node_projection(self.dropout(best_nodes)))

        has_answer_input = torch.cat([
            qsummary,
            best_nodes_proj,
            qsummary * best_nodes_proj,
            torch.abs(qsummary - best_nodes_proj)
        ], dim=-1)

        global_answer_score = self.global_answer_score(self.dropout(has_answer_input)).squeeze(1)

        # Return log(E[F1]) plus a small constant to avoid negative infinity...
        return out_soft, torch.log(expected_val + 1e-4), global_answer_score
