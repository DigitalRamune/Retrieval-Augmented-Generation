from .index_list import Chunk
import re


def clean_text(chunk_list: list[Chunk]) -> None:
    for chunk in chunk_list:
        lower_text = chunk.text.lower()
        clean_text = re.sub(r'[^\w\s]', ' ', lower_text)
        chunk.tokens = clean_text.split()


def create_vocab(chunk_list: list[Chunk]) -> dict[str, int]:
    global_vocab: dict[str, int] = {}
    word_index = 0
    for chunk in chunk_list: 
        for word in chunk.tokens:
            if word not in global_vocab:
                global_vocab[word] = word_index
                word_index += 1
    return global_vocab
