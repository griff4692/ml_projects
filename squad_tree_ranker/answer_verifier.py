import os

from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import contains_answer

options_file = os.path.join(os.getcwd(), 'data/elmo/options.json')
weight_file = os.path.join(os.getcwd(), 'data/elmo/weights.hdf5')

NEG_INF = -1e7


class AnswerVerifier(nn.Module):
    def __init__(self, model_dim=300, use_cuda=False, dropout_p=0.3):
        super(AnswerVerifier, self).__init__()
        self.use_cuda = use_cuda
        self.model_dim = model_dim

        # Set up embeddings as not trainable
        self.embeddings = Elmo(options_file, weight_file, 2, dropout=0, requires_grad=False)

        # Answer verification weights
        self.has_answer_encoder = nn.LSTM(input_size=2048 + 1, hidden_size=self.model_dim, batch_first=True,
                                          bidirectional=True, num_layers=1)

        self.has_answer_modelling = nn.LSTM(input_size=self.model_dim * 4, hidden_size=self.model_dim, batch_first=True,
                                            bidirectional=True, num_layers=1)

        self.fuser_g = nn.Linear(self.model_dim * 8, self.model_dim * 2, bias=False)
        self.fuser_r = nn.Linear(self.model_dim * 8, 1, bias=False)

        self.entailment_projection = nn.Linear(self.model_dim * 4, self.model_dim * 4, bias=True)
        self.has_answer_prediction = nn.Linear(self.model_dim * 4, 1, bias=True)

        self.alpha = self.cudify(nn.Parameter(torch.randn([1]), requires_grad=True))

        # Shared functions / operators
        self.softmax= nn.Softmax(dim=-1)
        self.softmax_row_wise = nn.Softmax(dim=-2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout_p)

    def _feature_learning(self, word_matrix):
        """
            Combines word level and character level embeddings by concatenation
        """

        # Convert to tensors
        elmo = self.embeddings(self.cudify(word_matrix))
        word_embed = torch.cat(elmo['elmo_representations'], dim=2)

        return word_embed

    def _fuser(self, x, y):
        feat = torch.cat([x, y, x * y, x - y], dim=-1)
        feat_dropout = self.dropout(feat)
        r = self.elu(self.fuser_r(feat_dropout))
        g = self.sigmoid(self.fuser_g(feat_dropout))

        return g * r + (1.0 - g) * x

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

    def _co_attn(self, x, y, attn_mask):
        sim = torch.bmm(x, y.transpose(1, 2))
        sim.data.masked_fill_(attn_mask, NEG_INF)

        x2y_weights = self.softmax(sim)
        y2x_weights = self.softmax_row_wise(sim)

        x2y = torch.bmm(x2y_weights, y)
        y2x = torch.bmm(y2x_weights.transpose(1, 2), x)

        return x2y, y2x

    def cudify(self, tensor):
        if self.use_cuda:
            return tensor.cuda()
        return tensor

    def forward(self, batch, predicted_node_idxs=None, eval_system=False):
        # self.has_answer_encoder.flatten_parameters(); self.has_answer_encoder.flatten_parameters()

        q_sizes = [len(batch.questions[i]) for i in range(batch.batch_size())]

        # Get question embeddings
        question = self._feature_learning(batch_to_ids(batch.questions))

        if eval_system:
            best_sentence_tokens, sentence_token_spans = batch.sentence_tokens_from_node_idxs(predicted_node_idxs)
        else:
            # Get embeddings only for sentence which contains (or most contains) answer
            best_sentence_tokens, sentence_token_spans = batch.context_best_sentence_tokens()

        context = self._feature_learning(batch_to_ids(best_sentence_tokens))

        best_sentence_sizes = [s[1] - s[0] for s in sentence_token_spans]
        max_sentence_size = context.size()[1]

        # augment question and context with binary indication if in answer or not
        c_contains_ans_onehot = np.zeros([batch.batch_size(), max_sentence_size, 1])
        q_contains_ans_onehot = np.zeros([batch.batch_size(), batch.max_question_size(), 1])
        for batch_idx in range(batch.batch_size()):
            answer = batch.predicted_answers[batch_idx] if eval_system else batch.answers[batch_idx]
            s = sentence_token_spans[batch_idx]
            sentence_tokens = batch.trees[batch.batch_idx_to_tree_idx(batch_idx)].tokens[s[0]: s[1]]

            c_binary_contains = [1.0 if contains_answer(t, answer) else 0.0 for t in sentence_tokens]
            c_contains_ans_onehot[batch_idx, :len(c_binary_contains), 0] = c_binary_contains

            q_binary_contains = [1.0 if contains_answer(t, answer) else 0.0 for t in batch.questions[batch_idx]]
            q_contains_ans_onehot[batch_idx, :len(q_binary_contains), 0] = q_binary_contains

        c_sentence_embed_expanded = torch.cat([
            context,
            self.cudify(Variable(torch.FloatTensor(c_contains_ans_onehot)))
        ], dim=-1)

        q_embed_expanded = torch.cat([
            question,
            self.cudify(Variable(torch.FloatTensor(q_contains_ans_onehot)))
        ], dim=-1)

        c_verifier_encoding, _ = self.has_answer_encoder(c_sentence_embed_expanded)
        q_verifier_encoding, _ = self.has_answer_encoder(q_embed_expanded)

        coattention_mask = self._attn_mask(batch, row_sizes=best_sentence_sizes, col_sizes=q_sizes, is_self=False)
        c2q, q2c = self._co_attn(c_verifier_encoding, q_verifier_encoding, coattention_mask)

        c_coattention_fused = self._fuser(c_verifier_encoding, c2q)
        q_coattention_fused = self._fuser(q_verifier_encoding, q2c)

        c_self_attention_mask = self._attn_mask(
            batch, row_sizes=best_sentence_sizes, col_sizes=best_sentence_sizes, is_self=True)
        q_self_attention_mask = self._attn_mask(
            batch, row_sizes=q_sizes, col_sizes=q_sizes, is_self=True)

        c2c, _ = self._co_attn(c_verifier_encoding, c_verifier_encoding, c_self_attention_mask)
        q2q, _ = self._co_attn(q_verifier_encoding, q_verifier_encoding, q_self_attention_mask)

        c_self_attention_fused = self._fuser(c_verifier_encoding, c2c)
        q_self_attention_fused = self._fuser(q_verifier_encoding, q2q)

        c_final_rep = self.dropout(torch.cat([c_coattention_fused, c_self_attention_fused], dim=-1))
        c_ans_verifier, _ = self.has_answer_modelling(c_final_rep)

        q_final_rep = self.dropout(torch.cat([q_coattention_fused, q_self_attention_fused], dim=-1))
        q_ans_verifier, _ = self.has_answer_modelling(q_final_rep)

        alpha = self.sigmoid(self.alpha)  # how much max versus mean pooling to use
        c_pooled = alpha * c_ans_verifier.max(dim=1)[0] + (1.0 - alpha) * c_ans_verifier.mean(dim=1)
        q_pooled = alpha * q_ans_verifier.max(dim=1)[0] + (1.0 - alpha) * q_ans_verifier.mean(dim=1)

        projection_input = self.dropout(torch.cat([c_pooled, q_pooled], dim=-1))
        projected = self.elu(self.entailment_projection(projection_input))
        answer_input = self.dropout(projected)
        answer_score = self.has_answer_prediction(answer_input).squeeze(1)

        return answer_score
