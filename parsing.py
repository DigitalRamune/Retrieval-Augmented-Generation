import fire
import os
import json
from tqdm import tqdm
from pathlib import Path
from retrieve_data import load_data, vectorize_query, compare_scores
from index_list import get_files_for_index, chunk_files
from term_frequency import get_tf_matrix, get_idf, get_tfidf, save_matrix
from vocab import clean_text, create_vocab


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
        return

    def index(self, max_chunk_size: int = 200,
              raw_dir: str = "data/raw/",
              processed_dir: str = "data/processed/") -> None:
        if max_chunk_size <= 0:
            print(f"Error: {max_chunk_size} is not a valid size.\n")
            return
        if not os.path.exists(raw_dir):
            print(f"Error: {raw_dir} is not a valid path.\n")
            return
        print("1. Recherche des fichiers...")
        files_list = get_files_for_index(raw_dir)
        print(f"2. Découpage de {len(files_list)} fichiers...")
        chunk_list = chunk_files(files_list, max_chunk_size)
        print(f"3. Nettoyage de {len(chunk_list)} chunks...")
        clean_text(chunk_list)
        print("4. Création du vocabulaire...")
        global_vocab = create_vocab(chunk_list)
        print(
            f"5. Matrice TF ({len(chunk_list)}x{len(global_vocab)})..."
        )
        tf_matrix = get_tf_matrix(chunk_list, global_vocab)
        idf_matrix = get_idf(tf_matrix)
        tfidf_matrix = get_tfidf(tf_matrix, idf_matrix)
        save_matrix(tfidf_matrix, global_vocab, chunk_list, processed_dir)


if __name__ == "__main__":
    fire.Fire(SearchEngineCLI)
