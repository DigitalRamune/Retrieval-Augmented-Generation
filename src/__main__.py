import fire
from typing import Any
import os
import json
from tqdm import tqdm
from pathlib import Path
from .retrieve_data import (load_data, vectorize_query, compare_scores,
                            build_context)
from .index_list import get_files_for_index, chunk_files
from .term_frequency import get_tf_matrix, get_idf, get_tfidf, save_matrix
from .vocab import clean_text, create_vocab
from .generator import generate_answer, create_tokenizer_model
from .data_models import (MinimalAnswer, StudentSearchResultsAndAnswer,
                          StudentSearchResults, MinimalSearchResults,
                          MinimalSource)


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
        for response in result:
            print(f"{response.file_path} "
                  f"[{response.first_character_index}:"
                  f"{response.last_character_index}]")

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
            try:
                os.makedirs(save_directory, exist_ok=True)
            except (OSError, PermissionError):
                print(f"Error: {save_directory} is not a directory.\n")
                return
        data = load_data(processed_dir)
        if not data:
            return
        matrix, vocab, chunks = data
        try:
            all_result = []
            writing_name = Path(dataset_path).name
            writing_path = os.path.join(save_directory,
                                        writing_name)
            with open(dataset_path, 'r', encoding='utf-8') as file:
                dataset = json.load(file)
                questions = (dataset.get("rag_questions", dataset)
                             if isinstance(dataset, dict) else dataset)
                for item in tqdm(questions):
                    query = item['question']
                    q_id = item.get("question_id", item.get("id", ""))
                    vector = vectorize_query(query, vocab)
                    result = compare_scores(vector, matrix, chunks, k)
                    if result is None:
                        result = []
                    new_result = MinimalSearchResults(
                        question_id=q_id,
                        question=query,
                        retrieved_sources=result)
                    all_result.append(new_result)
                result_obj = StudentSearchResults(search_results=all_result,
                                                  k=k)
            with open(writing_path, 'w', encoding='utf-8') as file:
                content = result_obj.model_dump_json(indent=4)
                file.write(content)

        except (FileNotFoundError, json.JSONDecodeError,
                OSError, PermissionError):
            print(f"Error: cannot read {dataset_path} file.\n")
            return
        return

    def index(self, max_chunk_size: int = 2000,
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

    def answer(self, query: str, k: int, processed_dir: str = "data/processed/") -> None:
        if k <= 0:
            print(f"Error: {k} is not a valid argument.\n")
            return
        data = load_data(processed_dir)
        if not data:
            return
        matrix, vocab, chunks = data
        vector = vectorize_query(query, vocab)
        result = compare_scores(vector, matrix, chunks, k)
        if not result:
            print("Error: list of results is empty.\n")
            return
        result_str = build_context(result)
        if not result_str:
            print("Error: cannot generate result string.\n")
            return
        final_answer = generate_answer(result_str, query)
        single_anwser = MinimalAnswer(question_id='1', question=query,
                                      retrieved_sources=result,
                                      answer=final_answer)
        output = StudentSearchResultsAndAnswer(search_results=[single_anwser],
                                               k=k)
        print(output.model_dump_json(indent=4))

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        if not student_search_results_path or not save_directory:
            print("Error: invalid parameter.\n")
            return
        result = Path(student_search_results_path)
        if not result.exists():
            print(f"Error: {student_search_results_path} is not a valid path."
                  "\n")
            return
        if not os.path.isdir(save_directory):
            try:
                os.makedirs(save_directory, exist_ok=True)
            except (PermissionError, OSError):
                print(f"Error: couldn't create {save_directory} directory.\n")
                return
        tokenizer, model = create_tokenizer_model()
        try:
            writing_name = Path(student_search_results_path).name
            writing_path = os.path.join(save_directory,
                                        writing_name)
            with open(student_search_results_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                k = data['k']
                answer_list: list[MinimalAnswer] = []
                search_list = data['search_results']
                for item in tqdm(search_list):
                    question_id = item.get('question_id', item.get('id', ''))
                    question = item.get('question')
                    sources = item.get("retrieved_sources")
                    minimal_list: list[MinimalSource] = []
                    for s in sources:
                        obj = MinimalSource(file_path=s["file_path"],
                                            first_character_index=int(
                                                s["first_character_index"]),
                                            last_character_index=int(
                                                s["last_character_index"]))
                        minimal_list.append(obj)
                    context = build_context(minimal_list)
                    answer = generate_answer(context=context,
                                             question=question,
                                             model=model,
                                             tokenizer=tokenizer)
                    new = MinimalAnswer(answer=answer,
                                        question_id=question_id,
                                        question=question,
                                        retrieved_sources=minimal_list)
                    answer_list.append(new)
                final_obj = StudentSearchResultsAndAnswer(
                    search_results=answer_list, k=k)
            with open(writing_path, 'w', encoding='utf-8') as file:
                content = final_obj.model_dump_json(indent=4)
                file.write(content)

        except (json.JSONDecodeError, PermissionError, TypeError):
            print("Error: cannot extract data from "
                  f"{student_search_results_path}.\n")
            return

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        if not student_search_results_path or not dataset_path:
            print("Error: invalid parameters.\n")
            return
        data_path = Path(dataset_path)
        stud_path = Path(student_search_results_path)
        if not data_path.exists():
            print(f"Error: {data_path} is not valid.\n")
            return
        if not stud_path.exists():
            print(f"Error: {stud_path} is not valid.\n")
            return
        try:
            with open(student_search_results_path, 'r', encoding='utf-8') as file:
                stud_content = json.load(file)
            with open(dataset_path, 'r', encoding='utf-8') as file:
                data_content = json.load(file)
            stud_prediction = stud_content.get('search_results')
            stud_k = stud_content.get('k')
            questions = (data_content.get("rag_questions", data_content)
                         if isinstance(data_content, dict) else data_content)
            ground_truth_dict: dict[Any, Any] = {}
            stud_k = int(stud_k)
            for item in tqdm(questions):
                question_id = item.get('question_id', item.get('id', ''))
                sources = item.get('sources', [])
                ground_truth_dict[question_id] = sources
            total_recall = 0.0
            valid_questions = 0
            for student_item in stud_prediction:
                q_id = student_item.get(
                    'question_id', student_item.get('id', ''))
                r_sources = student_item.get('retrieved_sources', [])
                r_sources = r_sources[:stud_k]
                gt_sources = ground_truth_dict.get(q_id, [])
                if not gt_sources:
                    continue
                hits = 0
                for gt in gt_sources:
                    for r in r_sources:
                        if ((gt["file_path"] == r["file_path"]) and
                            (max(gt['first_character_index'], r[
                                'first_character_index']) <
                             min(gt['last_character_index'], r[
                                 "last_character_index"]))):
                            hits += 1
                            break
                recall = hits / len(gt_sources)
                total_recall += recall
                valid_questions += 1
            if valid_questions > 0:
                average = total_recall / valid_questions
                print(f"Recall@{stud_k}: {average}")

        except (json.JSONDecodeError, PermissionError, IsADirectoryError):
            print("Error: cannot read files.\n")
            return


if __name__ == "__main__":
    fire.Fire(SearchEngineCLI)
