*This project has been created as part of the 42 curriculum by inaciri.*

# RAG against the machine

## Description
**RAG against the machine** is a command-line Retrieval-Augmented Generation (RAG) engine written in Python. Designed to answer technical questions about a large software repository (`vllm-0.10.1`), the system indexes source code and Markdown documentation, retrieves the most relevant character spans for a given query using a sparse BM25 lexical index, and feeds the bounded context to a local language model (`Qwen/Qwen3-0.6B`) to produce grounded, structured JSON answers.

---

## Instructions

### Prerequisites & Installation
This project uses `uv` for dependency and environment management. From the root of the repository, run:

```bash
uv sync
```
*(Or via the provided Makefile: `make install`)*

If working on a workstation with limited home directory disk quota, redirect `uv` and HuggingFace caches to an external volume before syncing:
```bash
export UV_CACHE_DIR=/path/to/external/.uv_cache
export HF_HOME=/path/to/external/.hf_cache
uv sync
```

### Code Quality & Linting
The codebase follows `flake8` and `mypy` static typing standards:
```bash
make lint
```

### Execution
All commands are exposed via Python Fire under the `src` module (`uv run python -m src <command> [options]`):
1. **Index the corpus:** `uv run python -m src index --max_chunk_size 2000`
2. **Search a dataset:** `uv run python -m src search_dataset --dataset_path <path> --k 10 --save_directory <dir>`
3. **Generate answers:** `uv run python -m src answer_dataset --student_search_results_path <path> --save_directory <dir>`
4. **Local evaluation:** `uv run python -m src evaluate --student_search_results_path <path> --dataset_path <path>`

---

## System Architecture
The system decouples offline corpus indexing from online retrieval and local LLM generation:

```text
Raw Corpus (data/raw/vllm-0.10.1)
       │
       ▼  [1. Ingestion & Chunking]
Segmented Chunks + Exact Character Offsets
       │
       ▼  [2. BM25 Indexing]
Sparse CSR Matrix (.npz) + Vocabulary & Metadata (.json) ──► data/processed/
       │
       ▼  [3. Lexical Retrieval]
Top-K Ranked MinimalSource Objects (file_path, start, end)
       │
       ▼  [4. Context Augmentation & Generation]
Bounded Context ──► Qwen/Qwen3-0.6B ──► Validated JSON Output
```

* **`src/__main__.py`:** Python Fire CLI entrypoint exposing `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, and `evaluate`, enforcing path validation, `tqdm` progress tracking, and exception handling.
* **`src/vocab.py` & `src/retrieve_data.py`:** Text preprocessing and tokenization pipeline.
* **`src/term_frequency.py`:** Vectorized NumPy/SciPy computation of BM25 term weights and IDF vectors stored in a Compressed Sparse Row (`csr_matrix`) format.
* **`src/generator.py`:** Local HuggingFace inference wrapper for `Qwen/Qwen3-0.6B`.
* **Pydantic Models:** Strictly typed schemas (`MinimalSource`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`) ensuring data integrity across pipeline stages and compatibility with the evaluation binary.

---

## Chunking Strategy
To balance retrieval granularity with the 2000-character `max_context_length` constraint enforced by the evaluation suite, documents in `data/raw/` are segmented according to file type (distinct handling for Python source files and Markdown/text documentation) into chunks bounded by `--max_chunk_size` (defaulting to `2000` characters).

Every chunk records its exact relative `file_path` alongside its `first_character_index` and `last_character_index`. Preserving exact byte/character offsets ensures verbatim path matching and accurate Intersection-over-Union (IoU > 0.05) scoring during evaluation.

---

## Retrieval Method
The retrieval engine implements **Okapi BM25** ($k_1 = 1.5$, $b = 0.75$) over a sparse document-term matrix:

1. **Smoothed Probabilistic IDF:**
   $$\text{IDF}(t) = \log\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)$$
   where $N$ is the total number of indexed chunks and $n(t)$ is the number of chunks containing term $t$.
2. **Term Frequency Saturation & Length Normalization:**
   $$\text{Score}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{\vert{}D\vert{}}{\text{avgdl}}\right)}$$
   where $f(t, D)$ is the raw term frequency in chunk $D$, $\vert{}D\vert{}$ is the token length of chunk $D$, and $\text{avgdl}$ is the average chunk length across the corpus.
3. **Fast Query Scoring:** Query tokens are mapped to a sparse query vector, and relevance scores across all chunks are computed via a single matrix-vector dot product (`matrix.dot(query_vector)`), returning the top-$k$ highest-scoring spans in milliseconds.

---

## Design Decisions
* **Preserving `snake_case` Identifiers in Tokenization:** Regex normalization uses `r'[^\w\s]'` rather than splitting on underscores (`_`). In code-heavy repositories like vLLM, exact function and parameter identifiers (e.g., `paged_attention`, `kv_cache_dtype`) carry high discriminative IDF weight; splitting them into generic sub-tokens introduces noise and degrades retrieval precision.
* **Sparse Matrix Architecture (`scipy.sparse.csr_matrix`):** Storing BM25 weights directly inside a CSR matrix keeps memory usage minimal and eliminates external retrieval library overhead while meeting the sub-90-second batch retrieval throughput requirement.
* **Single-Load LLM Inference:** In `answer_dataset`, the `Qwen/Qwen3-0.6B` model and tokenizer are instantiated once outside the iteration loop, and context windows are capped to prevent CPU RAM exhaustion and swap thrashing on campus workstations.
* **Pydantic Serialization:** All intermediate and final outputs are validated and dumped via Pydantic `.model_dump_json()` to guarantee strict adherence to the required JSON schema.

---

## Performance Analysis
Evaluated on the 100-question public documentation benchmark (`dataset_docs_public.json`, `max_chunk_size=2000`, `k=10`):

| Metric | Baseline (Raw TF-IDF) | Optimized (BM25) | Subject Threshold |
| :--- | :---: | :---: | :---: |
| **Recall@1** | 23.0% | **61.0%** | — |
| **Recall@3** | 42.0% | **77.0%** | — |
| **Recall@5** | 52.0% | **83.0%** | $\ge 80\%$ (Docs) / $\ge 50\%$ (Code) |
| **Recall@10** | 67.0% | **89.0%** | — |

* **Impact of BM25:** Upgrading from linear TF-IDF to BM25 boosted **Recall@5 by +31 percentage points** (52.0% $\rightarrow$ 83.0%) and **Recall@1 by +38 percentage points** (23.0% $\rightarrow$ 61.0%), surpassing the 80% documentation threshold. Linear TF-IDF systematically favored full 2000-character chunks over shorter, highly relevant snippets; BM25's document-length normalization ($b=0.75$) and term saturation ($k_1=1.5$) resolved this bias.
* **System Throughput:** Full corpus indexing completes well under the 5-minute limit, and batch retrieval across 100 questions runs in a few seconds.

---

## Challenges Faced
1. **Tokenization Granularity on Code:** Splitting underscores (`_`) during text cleaning initially dropped Recall@5 from 52% to 42% because compound variable names fractured into common, low-IDF words. Reverting to `r'[^\w\s]'` preserved exact identifier matching.
2. **CPU Memory Bottlenecks During Generation:** Passing 10 full chunks (up to 20,000 characters) per question to `Qwen/Qwen3-0.6B` on CPU-only machines caused quadratic attention memory spikes and disk swapping. Limiting the augmented context to top-ranked sources and bounding `max_context_length` stabilized RAM usage and prevented crashes.
3. **Filesystem Limitations on External Storage:** Developing on a FAT32/exFAT USB drive caused `uv sync` symlink failures (`Operation not permitted`), while default `uv` and HuggingFace caches filled local home quotas. Reformatting the drive to `ext4` and redirecting `UV_CACHE_DIR` and `HF_HOME` resolved both storage and symlink issues.

---

## Example Usage

```bash
# 1. Build the BM25 index from data/raw/
uv run python -m src index --max_chunk_size 2000

# 2. Query a single question from the CLI
uv run python -m src search "How to configure the OpenAI server?" --k 5
uv run python -m src answer "How to configure the OpenAI server?" --k 3

# 3. Run batch retrieval on a dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# 4. Evaluate retrieval recall@k with the official moulinette
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000

# 5. Generate grounded answers for the dataset
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

---

## Resources

### Technical References
* **SciPy Sparse Matrices:** [SciPy `csr_matrix` Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)
* **Pydantic V2:** [Pydantic Data Validation Documentation](https://docs.pydantic.dev/latest/)
* **Python Fire:** [Google Python Fire CLI Guide](https://github.com/google/python-fire)
* **Qwen3-0.6B:** [HuggingFace Model Card (`Qwen/Qwen3-0.6B`)](https://huggingface.co/Qwen/Qwen3-0.6B)
* **uv Package Manager:** [Astral `uv` Documentation](https://docs.astral.sh/uv/)

### AI Usage Disclosure
In accordance with the 42 AI guidelines, an AI assistant was used during development as a thought partner and debugging aid for the following specific tasks:
* **Algorithmic Design & Math Verification:** Reviewing the vectorized NumPy/SciPy implementation of Okapi BM25 term-frequency saturation and length normalization inside `term_frequency.py`, as well as validating the character-interval overlap logic (`max(start) < min(end)`) for the local `evaluate` command.
* **Debugging & Edge-Case Analysis:** Diagnosing tokenization regressions on `snake_case` identifiers, identifying CPU RAM saturation causes during `Qwen/Qwen3-0.6B` batch inference, and resolving `uv` environment/cache path issues on external Linux storage.
* **Documentation:** Structuring and proofreading this `README.md` to ensure all required project sections and explanations were clearly articulated.

All code architecture, Pydantic models, CLI orchestration, and final implementations were written, tested, and verified by the author.