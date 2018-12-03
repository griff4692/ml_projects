import torch
import torch.nn as nn
import import_spinn
from spinn import SPINN
from torch.autograd import Variable
from actions import HeKaimingInitializer, LayerNormalization
from utils import cudify

class FakeClassifier(nn.Module):
    def __init__(self, args, vocab, meta_vocab):
        super(FakeClassifier, self).__init__()

        self.META_LABELS = ["CATEGORIES", "NAME", "TITLE", "STATE", "PARTY", "VENUE"]

        padding_idx = vocab.stoi['<pad>']
        self.args = args
        self.embed = nn.Embedding(len(vocab.stoi), self.args.embed_dim, padding_idx=padding_idx)
        self.meta_embed = nn.Embedding(meta_vocab.size(), self.args.meta_embed_dim)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        self.layer_norm_mlp_input = LayerNormalization(self.args.hidden_size + 20)
        self.layer_norm_mlp1_hidden = LayerNormalization(self.args.classifier_dim)

        self.dropout = nn.Dropout(p=self.args.dropout_rate_classify)

        self.mlp1 = nn.Linear(self.args.hidden_size + 20, self.args.classifier_dim)
        HeKaimingInitializer(self.mlp1.weight)

        self.output = nn.Linear(self.args.classifier_dim, 6)
        HeKaimingInitializer(self.output.weight)
        self.spinn = SPINN(self.args)

        kernel_size = 3
        out_channels = 5
        self.cnn = torch.nn.Conv1d(len(self.META_LABELS), out_channels, kernel_size, stride=1)
        self.pool = nn.MaxPool1d(kernel_size, stride=1)

    def set_weight(self, weight):
        self.embed.weight.data.copy_(torch.from_numpy(weight))
        self.embed.weight.requires_grad = False

    def get_meta_mat(self, batch_meta):
        batch_meta_ids = []
        batch_cat_ids = []
        batch_cat_offsets = [0]
        for meta in batch_meta:
            for l in self.META_LABELS:
                if l == "CATEGORIES":
                    batch_cat_ids += meta[l]
                    batch_cat_offsets.append(len(meta[l]) + batch_cat_offsets[-1])
                else:
                    batch_meta_ids.append(meta[l])

        batch_meta_ids = cudify(self.args, Variable(torch.LongTensor(batch_meta_ids)))
        batch_meta_embeds = self.meta_embed(batch_meta_ids).view(len(batch_meta), len(self.META_LABELS) - 1, -1)

        batch_cat_ids = cudify(self.args, Variable(torch.LongTensor(batch_cat_ids)))
        batch_cat_embeds = self.meta_embed(batch_cat_ids)
        batch_avg_cat_embeds = []
        for b_id in range(len(batch_meta)):
            s, e = batch_cat_offsets[b_id], batch_cat_offsets[b_id + 1]
            cat_embeds = batch_cat_embeds[s:e, :]
            avg_embed = cat_embeds.mean(dim=0).unsqueeze(0)
            batch_avg_cat_embeds.append(avg_embed)

        batch_avg_cat_embeds = torch.cat(batch_avg_cat_embeds).unsqueeze(1)
        full_meta = torch.cat([batch_avg_cat_embeds, batch_meta_embeds], dim=1)
        return full_meta

    def extract_meta_features(self, meta_mat):
        meta_out = self.relu(self.cnn(meta_mat))
        meta_out_pool = self.pool(meta_out)
        return meta_out_pool.view(meta_out_pool.size()[0], -1)

    def build_biases(self, batch_meta):
        biases = []
        for meta in batch_meta:
            credit_vec = meta['CREDIT_VEC']
            credit_vec_sum = sum(credit_vec)

            for i in range(len(credit_vec)):
                if credit_vec_sum > 0:
                    credit_vec[i] /= float(credit_vec_sum)
                else:
                    credit_vec[i] = 1.0 / 6.0

            biases.append(cudify(self.args, Variable(torch.FloatTensor(credit_vec), requires_grad=False)).unsqueeze(0))
        return torch.cat(biases)

    def forward(self, sentence, meta, teacher_prob):
        meta_mat = self.get_meta_mat(meta)
        meta_feat = self.extract_meta_features(meta_mat)

        out_biases = self.build_biases(meta)

        embed = self.embed(sentence[0])
        trans = sentence[1]
        if self.args.tracking and (not self.training or not self.args.teacher):
            trans = None

        encode, sent_true, sent_pred = self.spinn(embed, trans, sentence[2], None, teacher_prob)

        mlp_input = torch.cat([encode, meta_feat], dim=-1)
        features = self.layer_norm_mlp_input(mlp_input)
        if self.args.dropout_rate_classify > 0:
            features = self.dropout(features)

        # ReLu plus weight matrix
        features = self.relu(self.mlp1(features))
        features = self.layer_norm_mlp1_hidden(features)

        if self.args.dropout_rate_classify > 0:
            features = self.dropout(features)

        output = self.output(features) + out_biases
        return output, sent_true, sent_pred
