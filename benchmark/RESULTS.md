# LongBench MultiFieldQA-en pilot results

A seeded sample of 10 examples from the 122 contexts containing at least 2,000 words. Sample context lengths: 2,076–7,551 words (median 2,831). Generator: local `llama3.2:latest` (Ollama model ID `a80c4f17acd5`); embedding model `nomic-embed-text:latest` (ID `0a109f422b47`).

| Method | Mean answer token F1 | Mean words passed to generator | Mean model calls | Median generator time |
|---|---:|---:|---:|---:|
| BM25 | 28.67% | 2,037 | 1.0 | 11.52 s |
| Dense cosine | 29.75% | 2,040 | 1.0 | 8.67 s |
| Dense/lexical hybrid | 26.45% | 2,037 | 1.0 | 5.68 s |
| RBER-style boundary peeling | 27.21% | 1,837 | 3.4 | 13.41 s |

Token F1 is overlap with the best reference after LongBench-style normalization. It is not exact match, evidence precision, or factuality. Three of ten baseline contexts contained an exact normalized reference answer string for each retrieval method; this is a very weak proxy for evidence recall, particularly for reference answers that summarize the text. NOT_FOUND responses were 4/10, 4/10, 5/10, and 5/10 respectively. Results are too small to establish a ranking or statistical significance.

The baseline methods each use at most eight overlapping passages and one model call. The RBER-style method uses the current backend boundary evaluator and boolean early stopping over recursively exposed thirds, but operates directly on original words; it does not run the backend's reconstruction, relevance-score bookkeeping, or HTTP layer. The paper's confidence threshold is represented by the implementation's binary response heuristic, not a calibrated probability. Prompts, context windows, and output limits differ between the RBER-style and baseline conditions. Fixed execution order and caching make measured times especially unsuitable for speed conclusions. Context words include repeated overlaps and calls, not model tokens. These measurements omit indexing, embedding, retrieval scoring, and network/service overhead outside calls. No full-context baseline, full application evaluation, human evidence annotation, confidence interval, or independent statistical test was performed.

See README.md in this directory for data, checksum, methodology, and reproduction command. Full per-case predictions are in results.json.
