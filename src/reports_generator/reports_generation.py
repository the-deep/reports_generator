import warnings
from typing import List
from collections import defaultdict
import networkx as nx
from typing import List, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans

import torch

from transformers import pipeline, AutoModel, AutoTokenizer
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")

from .utils import preprocess_sentences, build_graph, get_n_words
from .pooling import Pooling

n_min_sentences_for_summarization = 2


class ReportsGenerator:
    def __init__(
        self,
        summarization_model_name: str = "csebuetnlp/mT5_multilingual_XLSum",
        sentence_embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        sentence_embedding_output_length: int = 384,
    ):
        """
        Args:
            - summarization_model_name: generic summarization model from HuggingFace Summarization pipeline
            ( https://huggingface.co/models?pipeline_tag=summarization&sort=downloads )

            - sentence_embedding_model_name: multilingual model, used to get the sentence embeddings
            ( https://huggingface.co/models?pipeline_tag=fill-mask&sort=downloads )

            - sentence_embedding_output_length: output length of 'sentence_embedding_output_length' embeddings
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.summarization_model = pipeline(
            "summarization",
            model=summarization_model_name,
            tokenizer=summarization_model_name,
            # device=self.device,
        )

        self.embeddings_model = AutoModel.from_pretrained(
            sentence_embedding_model_name
        ).to(self.device)
        self.embeddings_tokenizer = AutoTokenizer.from_pretrained(
            sentence_embedding_model_name
        )

        self.pool = Pooling(
            word_embedding_dimension=sentence_embedding_output_length,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=True,
            pooling_mode_mean_sqrt_len_tokens=False,
        )

    def _get_sentences_embeddings(self, original_sentences):
        """
        get all tweets embeddings, one embedding per sentence
        """
        cleaned_text = preprocess_sentences(original_sentences)

        inputs = self.embeddings_tokenizer(
            cleaned_text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=40,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            transformer_output = self.embeddings_model(
                inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
            ).last_hidden_state.cpu()

        pooled_output = (
            self.pool(
                {
                    "token_embeddings": transformer_output,
                    "attention_mask": inputs["attention_mask"],
                }
            )["sentence_embedding"]
            .detach()
            .numpy()
        )

        return pooled_output

    def _get_clusters(self, embeddings):
        """
        1 - Get embeddings of tweets
        2 - Data reduction algorithm: UMAP if we have too many sentences to cluster
        3 - HDBscan clustering
        """
        n_rows = embeddings.shape[0]
        if n_rows <= 100:
            n_clusters = (n_rows // 10) + 1
        elif n_rows <= 200:
            n_clusters = (n_rows // 15) + 1
        else:
            n_clusters = min(n_rows // 40, 10)

        if n_clusters == 1:
            return np.ones(n_rows)
        else:
            # Clustering
            clustering_algo = MiniBatchKMeans(n_clusters=n_clusters)
            clusters = clustering_algo.fit(embeddings)
            return clusters.labels_

    def _summarize_one_cluster(
        self,
        original_sentences_one_cluster: List[str],
        embeddings_one_cluster,
    ) -> str:
        """
        Get summary for each cluster
        1 - Compute cosine similarity
        2 - Build undirected graph based on similarity matrix between excerpts
        3 - Get top n reelvant sentences using the pagerank algorithm
        4 - Generate summary of kept sentences
        """
        if type(embeddings_one_cluster) is list:
            embeddings_one_cluster = np.stack(embeddings_one_cluster)

        cosine_similarity_matrix = cosine_similarity(
            embeddings_one_cluster, embeddings_one_cluster
        )

        # get pagerank score for each sentence
        try:
            graph = build_graph(cosine_similarity_matrix)
            pageranks = nx.pagerank(graph)
            scores = np.array(list(pageranks.values()))

        except Exception:  # no ranking if pagerank algorithm doesn't converge
            scores = np.ones(len(embeddings_one_cluster))

        # keep sentences with highest scores
        top_n_sentence_ids = np.argsort(scores)[::-1]
        ranked_sentences = " ".join(
            [original_sentences_one_cluster[id_tmp] for id_tmp in (top_n_sentence_ids)]
        )

        # set max cluster summary length
        n_words = get_n_words(ranked_sentences)
        # max_length_one_cluster = min(n_words // 2, 128)  # 10 < max words < 128
        min_length_one_cluster = min(n_words // 4, 56)

        # summarize selected sentences
        try:
            summarized_entries = self.summarization_model(
                ranked_sentences,
                min_length=min_length_one_cluster,
                # max_length=max_length_one_cluster,
                truncation=True,
            )[0]["summary_text"]
        except Exception:  # case where input is too short
            summarized_entries = ""

        return summarized_entries

    def _multiclusters_summarization(
        self,
        entries_as_sentences: List[str],
        entries_embeddings,
        cluster_labels: List[int],
    ) -> str:
        dict_grouped_excerpts = {
            cluster_id: defaultdict(list) for cluster_id in list(set(cluster_labels))
        }
        n_sentences = len(entries_as_sentences)

        # Group sentences, embeddings into respective clusters.
        for i in range(n_sentences):
            cluster_i_label = cluster_labels[i]
            dict_grouped_excerpts[cluster_i_label]["sentences"].append(
                entries_as_sentences[i]
            )
            dict_grouped_excerpts[cluster_i_label]["embeddings"].append(
                entries_embeddings[i]
            )

        # summarize each cluster.
        summarized_entries_per_cluster = []
        for one_cluster_specifics in dict_grouped_excerpts.values():
            n_sentences_one_cluster = len(one_cluster_specifics["sentences"])
            if n_sentences_one_cluster >= n_min_sentences_for_summarization:
                summarized_entries_per_cluster.append(
                    self._summarize_one_cluster(
                        one_cluster_specifics["sentences"],
                        one_cluster_specifics["embeddings"],
                    )
                )

        return summarized_entries_per_cluster

    def _summarization_iteration(self, entries: Union[str, List[str]]) -> List[str]:

        # Get embeddings
        if type(entries) is str:
            entries = sent_tokenize(entries)

        entries_embeddings = self._get_sentences_embeddings(entries)

        # Get clusters
        cluster_labels = self._get_clusters(entries_embeddings)
        n_clusters = len(list(set(cluster_labels)))

        if n_clusters == 1:

            summarized_entries = [
                self._summarize_one_cluster(entries, entries_embeddings)
            ]

        else:
            summarized_entries = self._multiclusters_summarization(
                entries_as_sentences=entries,
                entries_embeddings=entries_embeddings,
                cluster_labels=cluster_labels,
            )

        return summarized_entries

    def __call__(
        self,
        entries: Union[str, List[str]],
        max_iterations: int,
        # max_summary_length: int = 384,
    ) -> str:
        """
        Args:
            - entries: text to be summarized, either as a form of a list of sentences or paragraph.
            - max_iterations: int: maximum number of iterations to be performed while summarizing
            - max_summary_length: maximum length of the summary
        """

        assert (
            max_iterations is not int
        ), "'max_iterations' parameter must be an integer."
        assert max_iterations >= 1, "'max_iterations' parameter must >= 1."

        if type(entries) is list:
            entries_as_str = " ".join([str(one_entry) for one_entry in entries])
        elif type(entries) is str:
            entries_as_str = entries
        else:
            AssertionError(
                "argument 'entries' must be one of the types [str, List[str]]"
            )

        n_words = get_n_words(entries_as_str) + 1
        if n_words < 20:
            warnings.warn(
                f"Warning... The minimum number of words in the input is 20 but yours is shorter ({n_words}). No summary has been generated and the output is an empty string. Please provide a longer input text for a good quality summary."
            )

        if len(sent_tokenize(entries_as_str)) < 2:
            warnings.warn(
                "Warning... The minimum number of input sentences must be at least 2 but your input consists of only one sentence. No summary has been generated and the output is an empty string. Please provide at least one more sentence for a good quality summary."
            )

        """n_raw_text_words = get_n_words(entries_as_str)
        if n_raw_text_words < max_summary_length:
            max_summary_length = (n_raw_text_words // 2) - 1
            max_iterations = 1
            warnings.warn(
                f"Warning... The length of the original text is smaller than the maximum summary length, setting 'max_iterations' parameter to 1 and the 'max_summary_length' to {max_summary_length}."
            )"""

        summarized_text = self._summarization_iteration(entries_as_str)
        n_iterations = 1

        while (
            n_iterations
            < max_iterations
            # and get_n_words(summarized_text) > max_summary_length
        ):
            summarized_text = self._summarization_iteration(summarized_text)
            n_iterations += 1

        """if (
            n_iterations == max_iterations
            and get_n_words(summarized_text) > max_summary_length
        ):
            warnings.warn(
                "Warning... Maximum number of iterations reached but summarized text length is still longer than the max_length, returning the long summarized version."
            )"""

        return " ".join(summarized_text)
