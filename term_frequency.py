from index_list import Chunk
import numpy
import os
import json
import pickle


def get_tf_matrix(chunk_list: list[Chunk], global_vocab: dict[str, int]
                  ) -> numpy.ndarray:
    matrix = numpy.zeros((len(chunk_list), len(global_vocab)))
    for row_index, chunk in enumerate(chunk_list):
        for word in chunk.tokens:
            col_index = global_vocab[word]
            matrix[row_index, col_index] += 1
    return matrix


def get_idf(tf_matrix: numpy.ndarray) -> numpy.ndarray:
    rows = tf_matrix.shape[0]
    documented_frequency = (tf_matrix > 0).sum(axis=0)
    idf_matrix = numpy.log(rows / documented_frequency)
    return idf_matrix


def get_tfidf(tf_matrix: numpy.ndarray, idf_matrix: numpy.ndarray
              ) -> numpy.ndarray:
    tfidf_matrix = tf_matrix * idf_matrix
    return tfidf_matrix


def save_matrix(tfidf_matrix: numpy.ndarray, global_vocab: dict[str, int],
                chunk_list: list[Chunk], dir_path: str) -> None:
    try:
        os.makedirs(dir_path, exist_ok=True)
    except (PermissionError, OSError):
        print(f"\nMissing permission to create directory: {dir_path}\n")
        return
    m_path = os.path.join(dir_path, 'tf_idf.npy')
    v_path = os.path.join(dir_path, 'vocab.json')
    c_path = os.path.join(dir_path, 'chunk.pkl')
    numpy.save(m_path, tfidf_matrix)
    try:
        with open(v_path, "w", encoding='utf-8'
                  ) as file:
            json.dump(global_vocab, file, indent=4)
    except (PermissionError, OSError, IsADirectoryError, FileNotFoundError):
        print("\nError while creating vocab.json file \n")
        return
    try:
        with open(c_path, "wb"
                  ) as file:
            pickle.dump(chunk_list, file)
    except (PermissionError, OSError, IsADirectoryError, FileNotFoundError):
        print("\nError while creating chunk.pkl file \n")
        return
    return
