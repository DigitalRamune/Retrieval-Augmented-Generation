"""Module for loading indexed data and retrieving search results."""

import json
import os
import pickle
import re
from typing import Any
import numpy
from numpy.typing import NDArray
from scipy import sparse
from .data_models import MinimalSource
from .index_list import Chunk


def load_data(
    processed_dir: str,
) -> tuple[sparse.csr_matrix, dict[str, int], list[Chunk]] | None:
    """Load the TF-IDF sparse matrix, vocabulary, and chunks from disk."""
    try:
        m_path = os.path.join(processed_dir, "tf_idf.npz")
        v_path = os.path.join(processed_dir, "vocab.json")
        c_path = os.path.join(processed_dir, "chunk.pkl")
        matrix: sparse.csr_matrix = sparse.load_npz(m_path)
        with open(v_path, "r", encoding="utf-8") as file:
            vocab: dict[str, int] = json.load(file)
        with open(c_path, "rb") as file:
            chunks: list[Chunk] = pickle.load(file)
        return matrix, vocab, chunks
    except (ValueError, FileNotFoundError, IsADirectoryError, OSError):
        print("Error while reading the data files")
        return None


def clean_query(query: str) -> list[str]:
    """Normalize and tokenize the search query string."""
    lower_text = query.lower()
    cleaned_text = re.sub(r"[^\w\s]", " ", lower_text)
    splited_query = cleaned_text.split()
    return splited_query


def vectorize_query(
    query: str, global_vocab: dict[str, int]
) -> NDArray[numpy.float64]:
    """Convert a search query into a term frequency vector."""
    cleaned_query = clean_query(query)
    vector: NDArray[numpy.float64] = numpy.zeros(
        len(global_vocab), dtype=numpy.float64
    )
    for word in cleaned_query:
        if word in global_vocab:
            index = global_vocab[word]
            vector[index] += 1
    return vector


def compare_scores(
    vector: NDArray[Any],
    matrix: sparse.csr_matrix,
    chunk_list: list[Chunk],
    k: int,
) -> list[MinimalSource] | None:
    """Compute similarity scores and return the top-k search results."""
    scores: NDArray[Any] = numpy.asarray(matrix.dot(vector)).ravel()
    if scores.size == 0 or numpy.max(scores) == 0:
        return None

    list_of_result: list[MinimalSource] = []
    list_of_index = numpy.argsort(scores)[-k:][::-1]
    for index in list_of_index:
        best_chunk = chunk_list[int(index)]
        list_of_result.append(
            MinimalSource(
                file_path=best_chunk.path,
                first_character_index=best_chunk.start_index,
                last_character_index=best_chunk.end_index,
            )
        )
    return list_of_result