# RBER document workspace

A local React and FastAPI application for searching text-based PDFs, UTF-8 text, and Markdown using the paper in this repository. Documents and embeddings are stored in `data/documents.sqlite3`; uploaded originals are not retained. Ollama provides the embedding and answer models.

## 🎓 Coursework & Academic Submission

All required academic deliverables, worksheets, and project documentation for evaluation are organized in the **[`submission_docs/`](submission_docs/)** directory:
* **Directory Link:** [`submission_docs/`](submission_docs/)
* **Included Deliverables:** Academic worksheets, assignment submissions, and project reports accompanying this codebase.
* Refer to [`submission_docs/README.md`](submission_docs/README.md) for individual document descriptions:
  - Unit 1 Worksheet: [`RM_worksheet_UNIT_1_filled_Krish_Aggarwal.pdf`](submission_docs/RM_worksheet_UNIT_1_filled_Krish_Aggarwal.pdf)
  - Unit 2 Worksheet: [`RM_UNIT_2_COMPLETED_KRISH_AGGARWAL.pdf`](submission_docs/RM_UNIT_2_COMPLETED_KRISH_AGGARWAL.pdf)
  - Unit 3 Worksheet: [`RM_UNIT_3_COMPLETED_KRISH_AGGARWAL.pdf`](submission_docs/RM_UNIT_3_COMPLETED_KRISH_AGGARWAL.pdf)

## 📄 Academic Research Paper (Overleaf & LaTeX Ready)

The complete 6+ page academic research paper formatted in IEEE Conference style is available in the **[`latex_paper/`](latex_paper/)** directory:
* **Overleaf 1-Click ZIP:** [`latex_paper/RBER_Research_Paper_Overleaf.zip`](latex_paper/RBER_Research_Paper_Overleaf.zip) (Ready to drag-and-drop into Overleaf via *New Project* $\to$ *Upload Project*).
* **Master LaTeX Source:** [`latex_paper/main.tex`](latex_paper/main.tex)
* **BibTeX References:** [`latex_paper/references.bib`](latex_paper/references.bib) (23+ authentic citations)
* **Paper Details & Guide:** [`latex_paper/README.md`](latex_paper/README.md)
* **Included Components:** Abstract, Introduction, Burger Analogy Theory, Mathematical Formulation, Algorithms 1 & 2, System Architecture & Attention Decay TikZ Diagrams, 4 Comparison & Ablation Tables, Vector UI Mockup, and Empirical Benchmarks.

## 📊 Datasets & Empirical Benchmarks

The repository provides automated reproduction suites across both controlled stress-tests and canonical real-world long-context academic benchmarks:

### 1. Adversarial Needle-in-a-Haystack (NIAH) Corpus
* **File:** [`long.txt`](long.txt)
* **Scale:** 21,408 words ($\sim 28,000$ tokens), dense longitudinal historical text.
* **Evaluation Protocol:** A targeted semantic needle (`SECRET KEY: The combination to the safe is "AGI is Coming"`) is systematically swept across 21 relative depth positions ($0\%, 5\%, 10\%, \dots, 100\%$) to test *Lost-in-the-Middle* degradation.
* **Benchmark Script:** [`benchmarks/benchmark_retrieval.py`](benchmarks/benchmark_retrieval.py)
* **Result:** Vanilla 128k LLMs collapse to 0.0% Exact Match in the middle 40%–60% zone; RBER achieves **100.0% Exact Match uniformly** with an 88.8% evaluated token reduction.

### 2. Real-World LongBench Benchmark (ACL 2024)
* **Dataset Mirrors:** 
  - [`passage_retrieval_en.jsonl`](https://huggingface.co/datasets/Leooyii/longbench/resolve/main/passage_retrieval_en.jsonl): 200 multi-page long documents ($7,379$ to $11,350$ words each, median $9,132$ words / $\sim 12,200$ tokens).
  - [`multifieldqa_en.jsonl`](https://huggingface.co/datasets/Leooyii/longbench/resolve/main/multifieldqa_en.jsonl): 150 diverse multi-domain documents (academic papers, legal transcripts, literature).
* **Citation:** Bai et al., *"LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding"*, ACL 2024.
* **Evaluation Script:** [`benchmark/evaluate_longbench.py`](benchmark/evaluate_longbench.py)
* **Detailed Report:** Full per-trial logs, depth classifications, and token statistics in [`benchmark/REAL_DATASET_REPORT.md`](benchmark/REAL_DATASET_REPORT.md).
* **Key Finding:** Fixed Boundary screening suffers a 0.0% collapse across the middle 20%–80% depth zone; standard dense vector chunking achieves only 36.7% Top-1 accuracy due to vector dilution. RBER achieves **86.7% overall retrieval recall** and **81.3% in the deep middle**, slashing evaluated token volume by **76.4%** in $d = 1.8$ mean steps.

### Quick Dataset Replication
```bash
# 1. Run the CPU-only adversarial NIAH benchmark (21 depths)
python3 benchmarks/benchmark_retrieval.py

# 2. Download and evaluate the real-world LongBench benchmark (e.g. 30 samples)
curl -L -o data/passage_retrieval_en.jsonl https://huggingface.co/datasets/Leooyii/longbench/resolve/main/passage_retrieval_en.jsonl
python3 benchmark/evaluate_longbench.py 30
```

## Run locally

1. Install [Ollama](https://ollama.com/) and start its service. Pull the models:
   ```sh
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```
2. Start the backend from the project root:
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn backend.app:app --host 127.0.0.1 --port 8000
   ```
3. In another terminal:
   ```sh
   cd frontend
   npm install
   npm run dev
   ```
   Open the local URL printed by Vite. Add documents, wait for indexing, select them, and ask a question. Expand the cited passages to inspect the source. Uploads default to 100 MB. PDF page numbers reflect physical page order. Scanned/image-only PDFs require OCR before upload.

Configuration environment variables: `OLLAMA_URL` (default `http://localhost:11434`), `EMBED_MODEL` (`nomic-embed-text`), `GEN_MODEL` (`llama3.2`), `SCREENER_MODEL` (defaults to `GEN_MODEL`; set to e.g. `llama3.2:1b` for fast speculative screening), `PARALLEL_BOUNDARIES` (default `true`), `VECTOR_GUIDED_PEELING` (default `true`), `SEAM_PADDING` (default `0`), `RBER_DATA_DIR`, `MAX_UPLOAD_MB` (100), and comma-separated `CORS_ORIGINS`. Changing the embedding model requires re-uploading documents. The Vite development server proxies `/api` to port 8000. For deployment, serve the frontend build behind a reverse proxy routing `/api` to the backend; add authentication and deployment-specific access controls before exposing it to other users.

## Retrieval design and limits

Documents are indexed as page-aware, overlapping passages and reconstructed for search. The backend follows the boundary evaluator in `res.py`: it splits each region into thirds and evaluates a bounded view of the beginning and end of each outer third (evaluating Left and Right in parallel when enabled). If neither yields an answer, it peels inward toward the region exhibiting the highest semantic concentration (or the middle core) and recurses. The final core is evaluated when it is small enough. Existing embeddings provide max-pooled vector similarity scores for the boundary trace and guided peeling. One question is processed at a time per backend process.

This approach examines boundary views rather than every passage. It can miss an answer or accept an unsupported answer, especially when the model returns something other than `NOT_FOUND` for irrelevant text. The displayed boundary view is the context the model saw, not a verified quotation. Check the source, especially for consequential decisions. PDFs with complicated layouts, tables, or image-only content may extract poorly.

## Tests

```sh
python -m unittest discover -s tests
cd frontend && npm run build
```
