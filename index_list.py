import pathlib


class Chunk:
    def __init__(self, start_index: int, end_index: int, text: str,
                 path: str):
        self.start_index = start_index
        self.end_index = end_index
        self.text = text
        self.path = path
        self.tokens: list[str] = []


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


def chunk_files(files_list: list[str], max_chunk_size: int) -> list[Chunk]:
    list_of_chunks: list[Chunk] = []
    for file in files_list:
        if file.endswith('.md'):
            try:
                with open(file, encoding="utf-8") as op_file:
                    # Save the content of the file in a variable
                    entire_content = op_file.read()
                    if entire_content:
                        current_index = 0
                        while current_index < len(entire_content):
                            temporary_chunk = entire_content[
                                current_index: current_index + max_chunk_size]

                            if (temporary_chunk[len(temporary_chunk) - 1
                                                ] != '\n' and temporary_chunk[
                                                    len(temporary_chunk) - 1
                                                    ] != '.'):
                                chunck_end_index = temporary_chunk.rfind(
                                    '\n\n')
                                if chunck_end_index == -1:
                                    chunck_end_index = temporary_chunk.rfind(
                                        '\n')
                                    if chunck_end_index == -1:
                                        chunck_end_index = (
                                            temporary_chunk.rfind('.'))
                                        if chunck_end_index == -1:
                                            chunck_end_index = len(
                                                temporary_chunk) - 1
                                final_chunk = temporary_chunk[
                                    :chunck_end_index]
                                chunk_obj = Chunk(current_index,
                                                  (current_index +
                                                   chunck_end_index),
                                                  final_chunk,
                                                  file)
                                list_of_chunks.append(chunk_obj)
                                current_index += chunck_end_index

                            else:
                                chunk_obj = Chunk(current_index,
                                                  (current_index
                                                   + max_chunk_size),
                                                  temporary_chunk,
                                                  file
                                                  )
                                list_of_chunks.append(chunk_obj)
                                current_index += max_chunk_size

            except (FileNotFoundError, IsADirectoryError, ValueError):
                print(f"Error: with {file}\n")

        elif file.endswith('.py'):
            try:
                with open(file, encoding="utf-8") as op_file:
                    entire_content = op_file.read()
                    if entire_content:
                        current_index = 0
                        while current_index < len(entire_content):
                            temporary_chunk = entire_content[
                                current_index: current_index + max_chunk_size]

                            if (temporary_chunk[len(temporary_chunk) - 1
                                                ] != '\n' and temporary_chunk[
                                                    len(temporary_chunk) - 2
                                                    ] != '\n'):
                                chunck_end_index = temporary_chunk.rfind(
                                    '\n\n')
                                if chunck_end_index == -1:
                                    chunck_end_index = temporary_chunk.rfind(
                                        '\n')
                                    if chunck_end_index == -1:
                                        chunck_end_index = len(
                                            temporary_chunk) - 1
                                final_chunk = temporary_chunk[
                                    :chunck_end_index]
                                chunk_obj = Chunk(current_index, (
                                    current_index + chunck_end_index
                                    ), final_chunk, file)
                                list_of_chunks.append(chunk_obj)
                                current_index += chunck_end_index

                            else:
                                chunk_obj = Chunk(current_index, (
                                    current_index + max_chunk_size
                                    ), temporary_chunk, file)
                                list_of_chunks.append(chunk_obj)
                                current_index += max_chunk_size

            except (FileNotFoundError, IsADirectoryError):
                print(f"Error: with {file}\n")

    return list_of_chunks


if __name__ == "__main__":
    files = get_files_for_index("./vllm-0.10.1/")
