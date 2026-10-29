"""Module for computing and saving TF, IDF, and TF-IDF sparse matrices."""

import gc
import json
import os
import pickle
import numpy
from numpy.typing import NDArray
from scipy import sparse
from .index_list import Chunk


def get_tf_matrix(
    chunk_list: list[Chunk], global_vocab: dict[str, int]
) -> sparse.csr_matrix:
    """Compute the Term Frequency (TF) sparse matrix from a list of chunks."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []

    for row_index, chunk in enumerate(chunk_list):
        word_counts: dict[int, int] = {}
        for word in chunk.tokens:
            col_index = global_vocab[word]
            word_counts[col_index] = word_counts.get(col_index, 0) + 1

        for col_index, count in word_counts.items():
            rows.append(row_index)
            cols.append(col_index)
            data.append(count)

        chunk.tokens.clear()
        chunk.text = ""

    gc.collect()

    row_arr = numpy.array(rows, dtype=numpy.int32)
    col_arr = numpy.array(cols, dtype=numpy.int32)
    data_arr = numpy.array(data, dtype=numpy.float32)

    matrix = sparse.csr_matrix(
        (data_arr, (row_arr, col_arr)),
        shape=(len(chunk_list), len(global_vocab)),
        dtype=numpy.float32,
    )

    del rows, cols, data, row_arr, col_arr, data_arr
    gc.collect()

    return matrix


def get_idf(tf_matrix: sparse.csr_matrix) -> NDArray[numpy.float32]:
    """Compute the Inverse Document Frequency (IDF) vector from a TF matrix."""
    rows = tf_matrix.shape[0]
    cols = tf_matrix.shape[1]
    doc_freq = numpy.bincount(tf_matrix.indices, minlength=cols)
    idf_raw = numpy.log(rows / (doc_freq + 1e-10))
    idf_matrix: NDArray[numpy.float32] = idf_raw.astype(numpy.float32)
    return idf_matrix


def get_tfidf(
    tf_matrix: sparse.csr_matrix, idf_matrix: NDArray[numpy.float32]
) -> sparse.csr_matrix:
    """Multiply the TF sparse matrix in-place by the IDF vector."""
    tf_matrix.data *= idf_matrix[tf_matrix.indices]
    return tf_matrix


def save_matrix(
    tfidf_matrix: sparse.csr_matrix,
    global_vocab: dict[str, int],
    chunk_list: list[Chunk],
    dir_path: str,
) -> None:
    """Save the TF-IDF matrix, vocabulary, and chunks to disk."""
    try:
        os.makedirs(dir_path, exist_ok=True)
    except (PermissionError, OSError):
        print(f"\nMissing permission to create directory: {dir_path}\n")
        return

    m_path = os.path.join(dir_path, "tf_idf.npz")
    v_path = os.path.join(dir_path, "vocab.json")
    c_path = os.path.join(dir_path, "chunk.pkl")

    sparse.save_npz(m_path, tfidf_matrix)

    try:
        with open(v_path, "w", encoding="utf-8") as file:
            json.dump(global_vocab, file, indent=4)
    except (PermissionError, OSError, IsADirectoryError, FileNotFoundError):
        print("\nError while creating vocab.json file\n")
        return

    try:
        with open(c_path, "wb") as file:
            pickle.dump(chunk_list, file)
    except (PermissionError, OSError, IsADirectoryError, FileNotFoundError):
        print("\nError while creating chunk.pkl file\n")
        return