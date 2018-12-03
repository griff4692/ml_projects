PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


class Vocab:
    def __init__(self):
        self.t2i = {}
        self.i2t = []

        self.pad_idx = self.add(PAD_TOKEN)
        assert self.pad_idx == 0
        self.unk_idx = self.add(UNK_TOKEN)
        assert self.unk_idx == 1

    def add(self, token):
        if not token in self.t2i:
            self.t2i[token] = self.size()
            self.i2t.append(token)
        return self.t2i[token]

    def contains_token(self, token):
        return token in self.t2i

    def get_id(self, token):
        assert token == token.lower()
        if self.contains_token(token):
            return self.t2i[token]
        else:
            return self.t2i[UNK_TOKEN]

    def get_token(self, id):
        return self.i2t[id]

    def size(self):
        return len(self.i2t)
