import csv
import pickle
import json
import numpy as np
import re

UNK = "<UNK>"

def to_key(idx, val):
    return str(idx) + '->' + str(val)

class MetaVocab:
    def __init__(self):
        self.m2i = {UNK: 0}
        self.i2m = [UNK]
        self.unkIdx = 0

    def size(self):
        return len(self.i2m)

    def add(self, m):
        # "" will be treated as null
        if len(m) == 0:
            m = UNK
        if m not in self.m2i:
            self.m2i[m] = len(self.i2m)
            self.i2m.append(m)
            assert self.i2m.index(m) == self.m2i[m]
        return self.m2i[m]

    def get_meta(self, idx):
        if idx < 0 or idx >= len(self.i2m):
            raise Exception("Not possible!")
        else:
            return self.i2m[idx]

    def to_idx(self, m):
        if m not in self.m2i:
            m = UNK
        return self.m2i[m]


PREPRO_LABEL_ORDER = ['half-true', 'false', 'mostly-true', 'true', 'barely-true', 'pants-fire']
DATASET_CREDIT_ORDER = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true']
REORDER = [PREPRO_LABEL_ORDER.index(l) for l in DATASET_CREDIT_ORDER]

def get_meta():
    vocab = MetaVocab()
    SERIALIZE = False
    flavors = ["train", "valid", "test"]
    meta_by_sent = {}

    for flavor in flavors:
        with open('liar_dataset/%s.tsv' % flavor, 'r') as csvin:
            reader = csv.reader(csvin, delimiter='\t')
            no = 0
            sent_lens = 0
            max_sent_len = 0
            max_num_cats = 0
            for row in reader:
                no += 1
                sentence = row[2]
                sent_len = len(sentence.split(" "))
                sent_lens += sent_len
                max_sent_len = max(sent_len, max_sent_len)
                meta = {}

                a = sentence.strip().lower()
                a = re.sub('[^A-Za-z0-9]+', '', a)

                # CATEGORIES
                cats = row[3].split(",")
                max_num_cats = max(len(cats), max_num_cats)
                meta['CATEGORIES'] = []

                # say category is unknown
                if len(cats) == 0:
                    meta['CATEGORIES'].append(vocab.to_idx(UNK))
                else:
                    for cat in cats:
                        if flavor == 'train':
                            meta['CATEGORIES'].append(vocab.add(cat))
                        else:
                            meta['CATEGORIES'].append(vocab.to_idx(cat))

                name = row[4]
                meta['NAME'] = vocab.add(name) if flavor == 'train' else vocab.to_idx(name)

                title = row[5]
                meta['TITLE'] = vocab.add(title) if flavor == 'train' else vocab.to_idx(title)

                state = row[6]
                meta['STATE'] = vocab.add(state) if flavor == 'train' else vocab.to_idx(state)

                party = row[7]
                meta['PARTY'] = vocab.add(party) if flavor == 'train' else vocab.to_idx(party)

                credit_vec = [int(score) for score in row[8:13]]
                reordered_credit_vec = [0] * len(PREPRO_LABEL_ORDER)
                for idx in range(len(credit_vec)):
                    reordered_credit_vec[REORDER[idx]] = credit_vec[idx]

                meta['CREDIT_VEC'] = reordered_credit_vec

                venue = row[13] if len(row) == 14 else ""
                meta['VENUE'] = vocab.add(venue) if flavor == 'train' else vocab.to_idx(venue)

                meta_by_sent[a] = meta

        print("%s Dataset size=%d" % (flavor, no))
        print("%s Average sentence length=%.2f.  (Max=%d).  (Max NumCats=%d)" % (flavor, float(sent_lens) / float(no), max_sent_len, max_num_cats))
        if SERIALIZE:
            out_meta = open('liar_dataset/%s_meta.json' % flavor, 'w')
            json.dump(meta_by_sent, out_meta)
    if SERIALIZE:
        out_dict = open('liar_dataset/meta_vocab.pk', 'wb')
        np.save(out_dict, vocab)

    return meta_by_sent, vocab

if __name__ == '__main__':
    get_meta()
