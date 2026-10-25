import fire
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


if __name__ == "__main__":
    fire.Fire(SearchEngineCLI)
