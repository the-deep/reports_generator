import warnings

warnings.filterwarnings("ignore")

from typing import List, Union
import networkx as nx
import re

# import nltk

# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")
# from nltk.corpus import stopwords

# stop_words = set(stopwords.words())


def _sentence_tokenize(sentence: str):
    return re.split(r"(?<=[^A-Z].[.?]) +(?=[A-Z])", sentence)


def flatten(t):
    """
    flatten a list of lists.
    """
    return [item for sublist in t for item in sublist]


def build_graph(cosine_similarity_matrix):
    """
    function to build graoh from similarity matrix
    """
    graph_one_lang = nx.Graph()
    matrix_shape = cosine_similarity_matrix.shape
    for i in range(matrix_shape[0]):
        for j in range(matrix_shape[1]):
            # do only once
            if i < j:
                sim = cosine_similarity_matrix[i, j]
                graph_one_lang.add_edge(i, j, weight=sim)
                graph_one_lang.add_edge(j, i, weight=sim)

    return graph_one_lang


def preprocess_one_sentence(sentence):
    """
    function to preprocess one_sentence:
        - lower and remove punctuation
        - remove stop words
        - stem and lemmatize
    """

    if type(sentence) is not str:
        sentence = str(sentence)

    return re.sub(r"[^\w\s]", "", sentence).rstrip().lstrip()


def preprocess_sentences(all_tweets: List[str]):
    """
    Clean list of sentences.
    """
    return [preprocess_one_sentence(one_tweet) for one_tweet in all_tweets]


def get_n_words_per_str(text: str):
    """
    Simple method to approximate the number of words in text.
    """
    return len(text.split(" "))


def get_n_words(entries: Union[str, List[str]]):
    """
    Simple method to approximate the number of words in text or list of texts.
    """
    if type(entries) is str:
        return get_n_words_per_str(entries)
    else:
        return sum([get_n_words_per_str(one_text) for one_text in entries])
