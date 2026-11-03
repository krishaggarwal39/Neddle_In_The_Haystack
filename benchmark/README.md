# LongBench pilot

The script evaluates the public LongBench MultiFieldQA-en 150-example file, using a fixed random seed (42), sampling from examples with at least 2,000 whitespace-separated context words. Data mirror: https://huggingface.co/datasets/Leooyii/longbench/resolve/main/multifieldqa_en.jsonl (SHA-256 `0aac182fd317dcf6d74f8e1e0f3e61029407435346c2e0b3ff9fb45ae49c5c3f`). The original dataset is described at https://huggingface.co/datasets/zai-org/LongBench and https://aclanthology.org/2024.acl-long.172/.

Run from the project root with the backend dependencies and Ollama models installed:

```
curl -L -o /tmp/multifieldqa_en.jsonl https://huggingface.co/datasets/Leooyii/longbench/resolve/main/multifieldqa_en.jsonl
python3 benchmark/run.py /tmp/multifieldqa_en.jsonl --n 10 --seed 42
```

Results are written after each completed example to `benchmark/results.json`. The full input corpus is not stored in this repository. Methods: BM25, dense cosine, and an experimental cosine/lexical hybrid each select up to eight overlapping backend chunks, ordered by position and passed to Llama 3.2. The RBER condition reproduces the current backend's ternary peeling, boundary-view evaluator, binary early stopping, and final core evaluation on the original context; it does not call the HTTP endpoint. It uses the application's evaluator prompt and limits. The other methods use a shorter common prompt and a different context limit, so this is an implementation-level comparison, not a controlled isolation of the algorithm alone. The methods do not have equal generator context budgets; RBER can make multiple calls.

Answer F1 uses lowercasing, punctuation and article removal, and token overlap against the best reference answer, as in LongBench-style English QA scoring. Exact answer-string presence in baseline context is a rough proxy only, not annotated evidence recall. Generator elapsed time excludes document embedding, query embedding, retrieval scoring, and indexing; runs are sequential and affected by model warm-up and caching. Word volume counts overlap and repeated model calls; it is not token usage or end-to-end latency. This is a small pilot, not a statistically reliable or leaderboard-comparable result. The subset excludes contexts shorter than 2,000 words. No manual grading, multiple seeds, external model comparisons, or long-context full-prompt baseline are included.
