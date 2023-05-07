from typing import *


class Vocabulary:
    def __init__(self, specials: List = None):
        self.w2i = {}
        self.i2w = {}
        self.idx = 0
        self.default_idx = None

        if specials:
            self.append_tokens(specials)

    def append_tokens(self, tokens):
        for token in tokens:
            self.append_token(token)

    def append_token(self, token):
        if token not in self.w2i:
            self.w2i[token] = self.idx
            self.i2w[self.idx] = token
            self.idx += 1

    def set_default_idx(self, default_idx):
        self.default_idx = default_idx

    def get_idxs(self, tokens):
        return [self[token] for token in tokens]

    def __contains__(self, token):
        return token in self.w2i

    def __getitem__(self, token):
        return self.w2i[token] if token in self.w2i else self.default_idx

    def __len__(self):
        return len(self.w2i)

    def __repr__(self) -> Any:
        return repr(self.w2i)
