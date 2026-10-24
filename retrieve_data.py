import numpy
import pickle
import json
import re
from index_list import Chunk


def load_data() -> tuple[numpy.ndarray, dict[str, int], list[Chunk]] | None:
    try:
        matrix = numpy.load('./data/processed/tf_idf.npy')
        with open('./data/processed/vocab.json', "r", encoding='utf-8'
                  ) as file:
            vocab = json.load(file)
        with open('./data/processed/chunk.pkl', "rb") as file:
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
