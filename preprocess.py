import datetime
import os
import re
import warnings
from typing import List

import emoji
import numpy as np
import py_vncorenlp

warnings.filterwarnings("ignore")

dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
}


def replace_all(text, dict_map):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text


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
        finally:
            if f is not None:
                f.close()

        self.segmenter = Segmenter(save_dir=vncorenlp_path, cur_dir=cur_dir)

    def remove_emoji(self, text: str) -> str:
        return emoji.replace_emoji(text, replace="")

    def remove_spaces(self, text: str) -> str:
        return re.sub(" +", " ", text)

    def remove_urls(self, text: str) -> str:
        return re.sub(r"\s*https?://\S+(\s+|$)", "", text).strip()

    def remove_stopwords(self, text: str) -> str:
        return " ".join(
            [token for token in text.split() if token not in self.stopwords]
        )

    def clean_one(self, text: str, astokens=False, dash=True) -> str:
        try:
            # lower text
            text = str(text).lower()

            # normalize Vietnamese
            text = replace_all(text, dict_map)

            # remove emojis
            text = self.remove_emoji(text)

            # remove url
            text = self.remove_urls(text)

            # remove punctuation
            text = re.sub(r"[^\w\s<>]", "", text)

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
        return [self.clean_one(text, astokens=astokens, dash=True) for text in texts]


def to_number(object):
    num = 0
    try:
        num = int(object)
    except:
        pass
    return num


def dict_handler(
    cleaner: VietnameseTextCleaner,
    dict_object: dict = None,
    islabeled=True,
):
    hours = datetime.datetime.fromtimestamp(dict_object["timestamp_post"]).hour
    weekdays = datetime.datetime.fromtimestamp(dict_object["timestamp_post"]).weekday()

    return {
        "id": dict_object["id"],
        "post_message": cleaner.clean_one(dict_object["post_message"]),
        "user_name": dict_object["user_name"],
        "metadata": np.log(
            np.array(
                (
                    to_number(dict_object["num_share_post"]),
                    to_number(dict_object["num_like_post"]),
                    to_number(dict_object["num_comment_post"]),
                    len(str(dict_object["post_message"]).split()),
                    weekdays,
                    hours,
                ),
                dtype=np.float32,
            )
            + 1
        ),
        "label": dict_object["label"],
    }
