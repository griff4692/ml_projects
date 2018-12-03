import json
from torchtext import datasets
from torchtext import data
import torchtext.vocab as vocab
from constants import PAD, SHIFT, REDUCE
import os

def prepare_fake_batches(args):
    data_dir = '.data/snli/snli_1.0'
    train_path = 'parsed_train.json'
    test_path = 'parsed_test.json'
    inputs = datasets.snli.ParsedTextField(lower=True)
    transitions = datasets.snli.ShiftReduceField()
    answers = data.Field(sequential=False)

    train, dev, test = datasets.SNLI.splits(inputs, answers, transitions, train=train_path, validation=test_path, test=test_path)

    inputs.build_vocab(train, dev, test)
    answers.build_vocab(train)
    glove = vocab.GloVe(name='840B', dim=args.embed_dim)
    inputs.vocab.set_vectors(stoi=glove.stoi, vectors=glove.vectors,dim=args.embed_dim)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=args.batch_size, device=args.gpu)

    return answers.vocab.itos, (train_iter, dev_iter, test_iter, inputs)
