import os
import py_vncorenlp
import emoji
from underthesea import text_normalize
import re
from typing import List
import warnings

warnings.filterwarnings("ignore")


class Segmenter(py_vncorenlp.VnCoreNLP):
    """
    Usage:

    segmenter = Segmenter()
    text = "Em không là nàng thơ. Anh cũng không còn là nhạc sĩ mộng mơ"
    segmenter.segment(text)
    >>> "Em không là nàng_thơ . Anh cũng không còn là nhạc_sĩ mộng_mơ"

    """

    def __init__(
        self,
        cur_dir,
        save_dir: str,
        max_heap_size="-Xms512m",
        annotators=["wseg"],
    ) -> None:
        super().__init__(max_heap_size, annotators, save_dir)
        os.chdir(cur_dir)

    def segment(self, text):
        return " ".join(self.word_segment(text))


class Cleaner:
    def __init__(self, cur_dir, stopwords_path: str, vncorenlp_path: str) -> None:
        f = None
        try:
            f = open(file=stopwords_path, mode="r", encoding="utf-8")
            self.stopwords = f.read().split()
        except Exception as e:
            print(e)
            print(stopwords_path)
        finally:
            if f is not None:
                f.close()

        self.segmenter = Segmenter(save_dir=vncorenlp_path, cur_dir=cur_dir)

    def remove_emoji(self, text: str) -> str:
        return emoji.replace_emoji(text, replace="")

    def normalize(self, text: str) -> str:
        return text_normalize(text)

    def remove_spaces(self, text: str) -> str:
        return re.sub(" +", " ", text)

    def remove_urls(self, text: str) -> str:
        return re.sub(r"\s*https?://\S+(\s+|$)", "", text).strip()

    def remove_stopwords(self, text: str) -> str:
        return " ".join(
            [token for token in text.split() if token not in self.stopwords]
        )

    def clean_one(self, text: str, astokens=False) -> str:
        try:
            text = text.lower()
            text = self.normalize(text)
            text = self.remove_emoji(text)
            text = self.remove_urls(text)
            text = self.segmenter.segment(text)
            text = self.remove_stopwords(text)
            return (
                text.replace("< url >", "<url>")
                if not astokens
                else text.replace("< url >", "<url>").split()
            )
        except Exception as e:
            print(text, e, type(text))

    def clean_many(self, texts: List, astokens=False) -> List:
        return [self.clean_one(text, astokens=astokens) for text in texts]
