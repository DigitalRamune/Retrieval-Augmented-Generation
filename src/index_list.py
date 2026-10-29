"""Module for discovering and chunking Markdown and Python files."""

import pathlib


class Chunk:
    """Represent a text segment extracted from a file."""

    def __init__(
        self, start_index: int, end_index: int, text: str, path: str
    ) -> None:
        """Initialize a Chunk instance with its boundaries and file path."""
        self.start_index = start_index
        self.end_index = end_index
        self.text = text
        self.path = path
        self.tokens: list[str] = []


def get_files_for_index(path: str) -> list[str]:
    """Retrieve all Python and Markdown file paths in the given directory."""
    dir_path = pathlib.Path(path)
    python_files = pathlib.Path.rglob(dir_path, "*.py")
    md_files = pathlib.Path.rglob(dir_path, "*.md")
    final_list: list[str] = []
    for file in python_files:
        final_list.append(str(file))
    for file in md_files:
        final_list.append(str(file))
    return final_list


def _find_split_index(
    temp_chunk: str, is_markdown: bool, max_chunk_size: int
) -> int:
    """Find the best character index to split a chunk without infinite loop."""
    if len(temp_chunk) < max_chunk_size:
        return len(temp_chunk)

    if is_markdown:
        if temp_chunk[-1] in ("\n", "."):
            return max_chunk_size
        separators: tuple[str, ...] = ("\n\n", "\n", ".")
    else:
        if temp_chunk.endswith("\n"):
            return max_chunk_size
        separators = ("\n\n", "\n")

    for sep in separators:
        idx = temp_chunk.rfind(sep)
        if idx > 0:
            return idx + len(sep)

    return max_chunk_size


def chunk_files(files_list: list[str], max_chunk_size: int) -> list[Chunk]:
    """Split the content of each file into a list of Chunk objects."""
    list_of_chunks: list[Chunk] = []

    for file in files_list:
        is_md = file.endswith(".md")
        is_py = file.endswith(".py")
        if not (is_md or is_py):
            continue

        try:
            with open(file, encoding="utf-8") as op_file:
                entire_content = op_file.read()
        except (
            FileNotFoundError,
            IsADirectoryError,
            ValueError,
            UnicodeDecodeError,
            OSError,
        ):
            print(f"Error: with {file}\n")
            continue

        if not entire_content:
            continue

        current_index = 0
        content_len = len(entire_content)

        while current_index < content_len:
            temporary_chunk = entire_content[
                current_index:current_index + max_chunk_size
            ]
            step = _find_split_index(temporary_chunk, is_md, max_chunk_size)
            final_chunk = temporary_chunk[:step]

            chunk_obj = Chunk(
                current_index,
                current_index + step,
                final_chunk,
                file,
            )
            list_of_chunks.append(chunk_obj)
            current_index += step

    return list_of_chunks


if __name__ == "__main__":
    files = get_files_for_index("./vllm-0.10.1/")
    