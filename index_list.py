import pathlib
from typing import Any


def get_files_for_index(path: str) -> list[str]:
    dir_path = pathlib.Path(path)
    python_files = pathlib.Path.rglob(dir_path, '*.py')
    md_files = pathlib.Path.rglob(dir_path, '*.md')
    final_list = []
    for file in python_files:
        final_list.append(str(file))
    for file in md_files:
        final_list.append(str(file))
    return final_list


def chunk_files(files_list: list[str], max_chunk_size: int) -> None:
    for file in files_list:
        if file.endswith('.md'):

        elif file.endswith('.py'):


if __name__ == "__main__":
    files = get_files_for_index("./vllm-0.10.1/")
