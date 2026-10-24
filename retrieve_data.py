import numpy
import pickle
import json
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
