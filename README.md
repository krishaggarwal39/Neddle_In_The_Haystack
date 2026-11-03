# RBER document workspace

A local React and FastAPI application for searching text-based PDFs, UTF-8 text, and Markdown using the paper in this repository. Documents and embeddings are stored in `data/documents.sqlite3`; uploaded originals are not retained. Ollama provides the embedding and answer models.

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
