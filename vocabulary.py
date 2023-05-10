from typing import *


class Vocabulary:
    def __init__(self, specials: List = None):
        self.w2i = {}
        self.i2w = {}
        self.idx = 0
        self.default_idx = None

        if specials:
            self.append_tokens(specials)

    def append_tokens(self, tokens: List):
        for token in tokens:
            self.append_token(token)

    def append_token(self, token: str):
        if token not in self.w2i:
            self.w2i[token] = self.idx
            self.i2w[self.idx] = token
            self.idx += 1

    def append_from_iterator(self, iterator: Iterable):
        for tokens in iterator:
            for token in tokens:
                try:
                    self.append_token(token)
                except Exception as e:
                    pass

    def set_default_idx(self, default_idx: int):
        self.default_idx = default_idx

    def get_idxs(self, tokens: List):
        return [self[token] for token in tokens]

    def __contains__(self, token: str):
        return token in self.w2i

    def __getitem__(self, token: str):
        return self.w2i[token] if token in self.w2i else self.default_idx

    def __len__(self):
        return len(self.w2i)

    def __repr__(self) -> Any:
        return repr(self.w2i)
