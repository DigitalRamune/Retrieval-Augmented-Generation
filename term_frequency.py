from index_list import Chunk
import numpy


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
