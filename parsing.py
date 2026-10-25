import fire
import os
import json
import pydantic
from tqdm import tqdm
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
        data = load_data(processed_dir)
        if not data:
            return
        matrix, vocab, chunks = data
        try:
            all_result = []
            writing_path = os.path.join(save_directory,
                                        'student_search_result.json')
            with open(dataset_path, 'r', encoding='utf-8') as file:
                dataset = json.load(file)
                for item in tqdm(dataset):
                    query = item["query"]
                    vector = vectorize_query(query, vocab)
                    result = compare_scores(vector, matrix, chunks, k)
                    if result is None:
                        result_dicts = []
                    else:
                        result_dicts = [r.model_dump() for r in result]
                    all_result.append({"id": item.get("id"),
                                       "query": query,
                                       "results": result_dicts})
            with open(writing_path, 'w', encoding='utf-8') as file:
                json.dump(all_result, file, indent=4)

        except (FileNotFoundError, json.JSONDecodeError,
                OSError, PermissionError):
            print(f"Error: cannot read {dataset_path} file.\n")
            return


if __name__ == "__main__":
    fire.Fire(SearchEngineCLI)
