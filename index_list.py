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
            try:
                with open(file, encoding="utf-8") as op_file:
                    entire_content = op_file.read()
                    if entire_content:
                        current_index = 0
                        while current_index < len(entire_content):
                            for char in entire_content:
                            


            except (FileNotFoundError, IsADirectoryError):
                print(f"Error: with {file}\n")

        elif file.endswith('.py'):
            try:
                with open(file, encoding="utf-8") as op_file:
                     entire_content = op_file.read()

            except (FileNotFoundError, IsADirectoryError):
                print(f"Error: with {file}\n")


if __name__ == "__main__":
    files = get_files_for_index("./vllm-0.10.1/")
