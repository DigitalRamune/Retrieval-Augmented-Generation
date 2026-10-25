import numpy
import pickle
import json
import re
import os
from index_list import Chunk
from data_models import SearchResult


def load_data(processed_dir: str) -> tuple[
     numpy.ndarray, dict[str, int], list[Chunk]] | None:
    try:
        m_path = os.path.join(processed_dir, 'tf_idf.npy')
        v_path = os.path.join(processed_dir, 'vocab.json')
        c_path = os.path.join(processed_dir, 'chunk.pkl')
        matrix = numpy.load(m_path)
        with open(v_path, "r", encoding='utf-8'
                  ) as file:
            vocab = json.load(file)
        with open(c_path, "rb") as file:
            chunks = pickle.load(file)
        return matrix, vocab, chunks
    except (ValueError, FileNotFoundError, IsADirectoryError, OSError):
        print("Error while reading the data files")
        return


def clean_query(query: str) -> list[str]:
    lower_text = query.lower()
    clean_text = re.sub(r'[^\w\s]', ' ', lower_text)
    splited_query = clean_text.split()
    return splited_query


def vectorize_query(query: str, global_vocab: dict[str, int]) -> numpy.ndarray:
    cleaned_query = clean_query(query)
    vector = numpy.zeros(len(global_vocab))
    for word in cleaned_query:
        if word in global_vocab:
            index = global_vocab[word]
            vector[index] += 1
    return vector


def compare_scores(vector: numpy.ndarray, matrix: numpy.ndarray,
                   chunk_list: list[Chunk],
                   k: int) -> list[SearchResult] | None:
    scores = matrix.dot(vector)
    if numpy.max(scores) == 0:
        return None
    list_of_result: list[SearchResult] = []
    list_of_index = numpy.argsort(scores)[-k:][::-1]
    for index in list_of_index:
        best_chunk = chunk_list[index]
        list_of_result.append(SearchResult(file_path=best_chunk.path,
                              first_character_index=best_chunk.start_index,
                              last_character_index=best_chunk.end_index))
    return list_of_result
