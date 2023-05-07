from preprocess import VietnameseTextCleaner
from tqdm.auto import tqdm


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
    # create a copy of object
    data = dict_object

    # preprocess
    data["num_share_post"] = to_number(data["num_share_post"])
    data["num_like_post"] = to_number(data["num_like_post"])
    data["num_comment_post"] = to_number(data["num_comment_post"])
    data["post_message"] = cleaner.clean_one(data["post_message"])
    return data


def LDtoDL(l):
    result = {}
    for d in l:
        for k, v in d.items():
            result[k] = result.get(k, []) + [v]
    return result


def DLtoLD(d):
    if not d:
        return []
    result = [{} for i in range(max(map(len, d.values())))]
    for k, seq in d.items():
        for oneDict, oneValue in zip(result, seq):
            oneDict[k] = oneValue
    return result

def generate_toy_dataset(length:int=100):
    pass