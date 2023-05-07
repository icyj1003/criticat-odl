import os
import py_vncorenlp
import emoji
from underthesea import text_normalize
import re
from typing import List
import warnings
from tqdm.auto import tqdm

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

    def __call__(self, text):
        return " ".join(self.word_segment(text))


class VietnameseTextCleaner:
    def __init__(self, cur_dir, stopwords_path: str, vncorenlp_path: str) -> None:
        """_summary_

        Args:
            cur_dir (_type_): _description_
            stopwords_path (str): _description_
            vncorenlp_path (str): _description_
        """
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
        """_summary_

        Args:
            text (str): _description_

        Returns:
            str: _description_
        """
        return emoji.replace_emoji(text, replace="")

    def normalize(self, text: str) -> str:
        """_summary_

        Args:
            text (str): _description_

        Returns:
            str: _description_
        """
        return text_normalize(text)

    def remove_spaces(self, text: str) -> str:
        """_summary_

        Args:
            text (str): _description_

        Returns:
            str: _description_
        """
        return re.sub(" +", " ", text)

    def remove_urls(self, text: str) -> str:
        """_summary_

        Args:
            text (str): _description_

        Returns:
            str: _description_
        """
        return re.sub(r"\s*https?://\S+(\s+|$)", "", text).strip()

    def remove_stopwords(self, text: str) -> str:
        """_summary_

        Args:
            text (str): _description_

        Returns:
            str: _description_
        """
        return " ".join(
            [token for token in text.split() if token not in self.stopwords]
        )

    def clean_one(self, text: str, astokens=False, dash=True) -> str:
        """_summary_

        Args:
            text (str): _description_
            astokens (bool, optional): _description_. Defaults to False.
            dash (bool, optional): _description_. Defaults to True.

        Returns:
            str: _description_
        """
        try:
            text = str(text).lower()
            text = self.normalize(text)
            text = self.remove_emoji(text)
            text = self.remove_urls(text)
            if dash:
                text = self.segmenter(text)
            text = self.remove_stopwords(text)
            return (
                text.replace("< url >", "<url>")
                if not astokens
                else text.replace("< url >", "<url>").split()
            )
        except Exception as e:
            print(text, e, type(text), str(text))

    def clean_many(self, texts: List, astokens=False, dash=True) -> List:
        """_summary_

        Args:
            texts (List): _description_
            astokens (bool, optional): _description_. Defaults to False.
            dash (bool, optional): _description_. Defaults to True.

        Returns:
            List: _description_
        """
        return [
            self.clean_one(text, astokens=astokens, dash=True) for text in tqdm(texts)
        ]


class SimpleTextCleaner:
    def __init__(self) -> None:
        self.tokenizer = lambda x: x.split()

    def clean_one(self, text):
        return self.tokenizer(re.match(r"[a-z, ]*", text.lower())[0])

    def clean_many(self, texts):
        return [self.clean_one(text) for text in texts]
