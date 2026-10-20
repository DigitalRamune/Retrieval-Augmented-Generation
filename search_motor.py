from index_list import Chunk
import re


def clean_text(chunk_list: list[Chunk]) -> list[str]:
    clean_list: list[str] = []
    for chunk in chunk_list:
        lower_text = chunk.text.lower()
        clean_text = re.sub(r'[^\w\s]', ' ', lower_text)
