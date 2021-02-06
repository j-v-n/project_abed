from nltk.tokenize import TweetTokenizer
import collections
import itertools
import pickle
import os
import logging
from tqdm import tqdm

DATA_DIR = "./data/"
UNKNOWN_TOKEN = "#UNK"
BEGIN_TOKEN = "#BEG"
END_TOKEN = "#END"
# names = ["test", "train"]

MAX_TOKENS = 30
MIN_TOKEN_FREQ = 10
SHUFFLE_SEED = 5871

EMB_DICT_NAME = "emb_dict.dat"
EMB_NAME = "emb.npy"
PHRASE_PAIRS_NAME = "phrase_pairs.dat"
log = logging.getLogger("data")


def save_emb_dict(dir_name, emb_dict):
    with open(os.path.join(dir_name, EMB_DICT_NAME), "wb") as fd:
        pickle.dump(emb_dict, fd)


def load_emb_dict(dir_name):
    with open(os.path.join(dir_name, EMB_DICT_NAME), "rb") as fd:
        return pickle.load(fd)


def save_phrase_pairs(dir_name, phrase_pairs):
    with open(os.path.join(dir_name, PHRASE_PAIRS_NAME), "wb") as fd:
        pickle.dump(phrase_pairs, fd)


def load_phrase_pairs(dir_name):
    with open(os.path.join(dir_name, PHRASE_PAIRS_NAME), "rb") as fd:
        return pickle.load(fd)


def tokenize(s):
    return TweetTokenizer(preserve_case=False).tokenize(s)


def generate_dialogues(name, data_dir=DATA_DIR):
    """
    Given a particular name (test or train), this function creates pairs of dialogues
    using the from and to files

    """

    from_list = []
    to_list = []
    with open(data_dir + name + ".from.new", "r") as from_file:
        for line in tqdm(from_file):
            tokenized_in_line = tokenize(line)
            from_list.append(tokenized_in_line)

    with open(data_dir + name + ".to.new", "r") as to_file:
        for line in tqdm(to_file):
            tokenized_out_line = tokenize(line)
            to_list.append(tokenized_out_line)

    dialogues = list(zip(from_list, to_list))

    return dialogues


def dialogues_to_pairs(dialogues, max_tokens):
    """
    Iterate through each pair of dialogues and check if the length of each phrase in a pair is less than max_tokens.
    If it is, create a tuple of parent and comment.
    Append to the list called results and return results
    """

    result = []
    for dial in tqdm(dialogues):
        prev_phrase = None
        for phrase in dial:
            if prev_phrase is not None:
                if max_tokens is None or (
                    len(prev_phrase) <= max_tokens and len(phrase) <= max_tokens
                ):
                    result.append((prev_phrase, phrase))

            prev_phrase = phrase
    return result


def phrase_pairs_dict(phrase_pairs, freq_set):
    """
    Iterate through every word in a phrase pair and check if it is in the list of frequently occuring words
    and if it is, then add it to a dictionary. The first three entries in the dictionary correspond to UNKNOWN_TOKEN,
    BEGIN_TOKEN and END_TOKEN
    """

    res = {UNKNOWN_TOKEN: 0, BEGIN_TOKEN: 1, END_TOKEN: 2}

    next_id = 3

    for p1, p2 in tqdm(phrase_pairs):
        for word in map(str.lower, itertools.chain(p1, p2)):
            if word not in res and word in freq_set:
                res[word] = next_id
                next_id += 1

    return res


def load_data(name, max_tokens, min_token_freq):
    dialogues = generate_dialogues(name=name)
    print("{} dialogues obtained".format(len(dialogues)))

    # convert list of dialogues to pairs
    phrase_pairs = dialogues_to_pairs(dialogues, max_tokens=max_tokens)
    word_counts = collections.Counter()

    for dial in tqdm(dialogues):
        for line in dial:
            word_counts.update(line)

    freq_set = set(
        map(
            lambda p: p[0],
            filter(lambda p: p[1] >= min_token_freq, word_counts.items()),
        )
    )

    # create a dictionary based on phrase pairs and
    phrase_dict = phrase_pairs_dict(phrase_pairs, freq_set)

    return phrase_pairs, phrase_dict


def iterate_batches(data, batch_size):
    assert isinstance(data, list)
    assert isinstance(batch_size, int)

    ofs = 0
    while True:
        batch = data[ofs * batch_size : (ofs + 1) * batch_size]
        if len(batch) <= 1:
            break
        yield batch
        ofs += 1


def decode_words(indices, rev_emb_dict):
    return [rev_emb_dict.get(idx, UNKNOWN_TOKEN) for idx in indices]


def trim_tokens_seq(tokens, end_token):
    res = []
    for t in tokens:
        res.append(t)
        if t == end_token:
            break
    return res


def encode_words(words, emb_dict):
    """
    Convert list of words into list of embeddings indices, adding our tokens
    :param words: list of strings
    :param emb_dict: embeddings dictionary
    :return: list of IDs
    """
    res = [emb_dict[BEGIN_TOKEN]]
    unk_idx = emb_dict[UNKNOWN_TOKEN]
    for w in words:
        idx = emb_dict.get(w.lower(), unk_idx)
        res.append(idx)
    res.append(emb_dict[END_TOKEN])
    return res


def encode_phrase_pairs(phrase_pairs, emb_dict, filter_unknows=True):
    """
    Convert list of phrase pairs to training data
    :param phrase_pairs: list of (phrase, phrase)
    :param emb_dict: embeddings dictionary (word -> id)
    :return: list of tuples ([input_id_seq], [output_id_seq])
    """
    unk_token = emb_dict[UNKNOWN_TOKEN]
    result = []
    for p1, p2 in phrase_pairs:
        p = encode_words(p1, emb_dict), encode_words(p2, emb_dict)
        if unk_token in p[0] or unk_token in p[1]:
            continue
        result.append(p)
    return result
