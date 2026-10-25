import fire
import os
from pathlib import Path
from retrieve_data import load_data, vectorize_query, compare_scores


class SearchEngineCLI:
    def search(self, query: str, k: int,
               processed_dir: str = "data/processed/") -> None:
        if not query or k <= 0:
            print("Error with the parameters")
            return
        data = load_data(processed_dir)
        if not data:
            return
        matrix, vocab, chunks = data
        vector = vectorize_query(query, vocab)
        result = compare_scores(vector, matrix, chunks, k)
        print(f"{result}\n")

    def search_dataset(self, dataset_path: str,
                       k: int, save_directory: str,
                       processed_dir: str = "data/processed/") -> None:
        if not dataset_path or k <= 0 or not save_directory:
            print("Error: Invalid parameters.\n")
            return
        path = Path(dataset_path)
        if not path.exists():
            print(f"Error: {dataset_path} is not a valid path.\n")
            return
        if not os.path.isdir(save_directory):
            print(f"Error: {save_directory} is not a directory.\n")
            return


if __name__ == "__main__":
    fire.Fire(SearchEngineCLI)
