import torch
import torch.nn as nn
import import_spinn
from spinn import SPINN
from torch.autograd import Variable
from actions import HeKaimingInitializer, LayerNormalization
from utils import cudify

class SNLIClassifier(nn.Module):
    def __init__(self, args, vocab):
        super(SNLIClassifier, self).__init__()

        padding_idx = vocab.stoi['<pad>']
        self.args = args
        self.embed = nn.Embedding(len(vocab.stoi), self.args.embed_dim, padding_idx=padding_idx)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        self.layer_norm_mlp_input = LayerNormalization(4 * self.args.hidden_size)
        self.layer_norm_mlp1_hidden = LayerNormalization(self.args.snli_h_dim)
        self.layer_norm_mlp2_hidden = LayerNormalization(self.args.snli_h_dim)

        self.dropout = nn.Dropout(p=self.args.dropout_rate_classify)

        self.mlp1 = nn.Linear(4 * self.args.hidden_size, self.args.snli_h_dim)
        HeKaimingInitializer(self.mlp1.weight)
        self.mlp2 = nn.Linear(self.args.snli_h_dim, self.args.snli_h_dim)
        HeKaimingInitializer(self.mlp2.weight)

        self.output = nn.Linear(self.args.snli_h_dim, 3)
        HeKaimingInitializer(self.output.weight)
        self.spinn = SPINN(self.args)

        self.encoder = nn.LSTM(input_size=self.args.embed_dim, hidden_size=self.args.hidden_size // 2,
            batch_first=True, bidirectional=False, num_layers=1, dropout=self.args.dropout_rate_input)
        self.init_lstm_state = cudify(args, Variable(torch.zeros(1, 1, self.args.hidden_size // 2), requires_grad=False))

    def encode_sent(self, sent):
        timesteps = sent.split(1, 1)
        n = len(timesteps)

        for_hs = [None] * n
        back_hs = [None] * n
        for_cs = [None] * n
        back_cs = [None] * n

        # forward pass
        init_h, init_c = self.init_lstm_state.repeat(1, sent.size()[0], 1), self.init_lstm_state.repeat(1, sent.size()[0], 1)
        prev_h, prev_c = init_h, init_c
        for i in range(n):
            _, (step_h, step_c) = self.encoder(timesteps[i], (prev_h, prev_c))
            for_hs[i] = step_h
            for_cs[i] = step_c
            prev_h, prev_c = step_h, step_c

        prev_h, prev_c = init_h, init_c
        for i in range(n - 1, -1 , -1):
            _, (step_h, step_c) = self.encoder(timesteps[i], (prev_h, prev_c))
            back_hs[i] = step_h
            back_cs[i] = step_c
            prev_h, prev_c = step_h, step_c

        for_hs, for_cs = torch.cat(for_hs), torch.cat(for_cs)
        back_hs, back_cs = torch.cat(back_hs), torch.cat(back_cs)

        full_hs = torch.cat([for_hs, back_hs], dim=2).transpose(0, 1)
        full_cs = torch.cat([for_cs, back_cs], dim=2).transpose(0, 1)

        last_for_hs, last_back_hs = for_hs[-1], back_hs[0]
        last_for_cs, last_back_cs = for_cs[-1], back_cs[0]

        summary_h = torch.cat([last_for_hs, last_back_hs], dim=1)
        summary_c = torch.cat([last_for_cs, last_back_cs], dim=1)
        return (summary_h, summary_c)

    def set_weight(self, weight):
        self.embed.weight.data.copy_(torch.from_numpy(weight))
        self.embed.weight.requires_grad = False

    def prepare_features(self, hyp, prem):
        return torch.cat([
            hyp, prem, prem - hyp,
            torch.mul(hyp, prem)
        ], dim=1)

    def forward(self, hypothesis, premise, teacher_prob):
        hyp_embed = self.embed(hypothesis[0])
        prem_embed = self.embed(premise[0])

        hyp_summary = self.encode_sent(hyp_embed)
        prem_summary = self.encode_sent(prem_embed)

        if not self.args.teacher or not self.training:
            hyp_trans, prem_trans = hypothesis[1], premise[1]
            if self.args.tracking:
                hyp_trans, prem_trans = None, None

            hyp_encode = self.spinn(hyp_embed, hyp_trans, hypothesis[2], prem_summary, teacher_prob)
            prem_encode = self.spinn(prem_embed, prem_trans, premise[2], hyp_summary, teacher_prob)
            sent_true, sent_pred = None, None
        else:
            hyp_encode, hyp_true, hyp_pred = self.spinn(hyp_embed, hypothesis[1], hypothesis[2], prem_summary, teacher_prob)
            prem_encode, prem_true, prem_pred = self.spinn(prem_embed, premise[1], premise[2], hyp_summary, teacher_prob)
            sent_true = torch.cat([hyp_true, prem_true])
            sent_pred = torch.cat([hyp_pred, prem_pred])

        features = self.prepare_features(hyp_encode, prem_encode)
        features = self.layer_norm_mlp_input(features)

        if self.args.dropout_rate_classify > 0:
            features = self.dropout(features)

        # ReLu plus weight matrix
        features = self.relu(self.mlp1(features))
        features = self.layer_norm_mlp1_hidden(features)

        # dropout
        if self.args.dropout_rate_classify > 0:
            features = self.dropout(features)

        features = self.relu(self.mlp2(features))
        features = self.layer_norm_mlp2_hidden(features)

        if self.args.dropout_rate_classify > 0:
            features = self.dropout(features)

        output = self.output(features)
        return output, sent_true, sent_pred
