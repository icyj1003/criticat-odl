from typing import *

import torch
import torchtext
from transformers import AutoModel, AutoTokenizer, logging

logging.set_verbosity(logging.CRITICAL)


class PhoBertVector:
    def __init__(
        self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:
        self.device = device
        self._model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    def get_vecs_by_raw_text(self, raw_text: str, max_length: int = 20) -> torch.Tensor:
        assert type(raw_text) is str, "raw_text must be a string"
        input_ids = torch.tensor(
            [
                self._tokenizer.encode(
                    raw_text,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                )
            ],
            device=self.device,
        )
        with torch.no_grad():
            features = self._model(input_ids)
        return features["last_hidden_state"].squeeze(0).cpu()


class PhoVector:
    def __init__(
        self, name: str = "word2vec_vi_words_100dims.txt", cache: str = "./.cache/"
    ) -> None:
        self._vectors = torchtext.vocab.Vectors(name=name, cache=cache)

    def get_vecs_by_raw_text(self, raw_text: str, max_length: int = 20) -> torch.Tensor:
        assert type(raw_text) is str, "raw_text must be a string"
        tokens = raw_text.split()
        if len(tokens) < max_length:
            pad_length = max_length - len(tokens)
            pad_tensor = torch.zeros((pad_length, 100))
            return torch.cat([self._vectors.get_vecs_by_tokens(tokens), pad_tensor])
        else:
            tokens = tokens[:20]
            return self._vectors.get_vecs_by_tokens(tokens)

    def get_vecs_by_tokens(self, tokens: list, max_length: int = 20) -> torch.Tensor:
        assert type(tokens) is list, "tokens must be a list"
        if len(tokens) < max_length:
            pad_length = max_length - len(tokens)
            pad_tensor = torch.zeros((pad_length, 100))
            return torch.cat([self._vectors.get_vecs_by_tokens(tokens), pad_tensor])
        else:
            tokens = tokens[:20]
            return self._vectors.get_vecs_by_tokens(tokens)


if __name__ == "__main__":
    v = PhoBertVector()
    a = v.get_vecs_by_raw_text(
        "thần_thánh , 10 họp cãi tiềm_lực trung_quốc hai bệnh_viện khổng_lồ sông 40 phút di_chuyển .... , tổng_công_ty doanh_nghiệp ,"
    )
    print(a.shape)
