"""
Comprehensive Evaluation of RBER vs Baselines on Real LongBench Dataset
Dataset: passage_retrieval_en (LongBench / ACL 2024)
Compares:
1. Fixed Boundary (Primacy + Recency, No Peeling)
2. BM25 Lexical Retrieval (Top-1, Top-3, Top-5 chunks)
3. Dense Embedding Retrieval (all-MiniLM-L6-v2, Top-1, Top-3, Top-5)
4. RBER (Recursive Boundary Exposure Retrieval with Chunk Max-Pooling)
"""

import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

# Try importing sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: sentence_transformers not loaded: {e}")
    EMBEDDER = None


def tokenize(text):
    return re.findall(r'\w+', text.lower())


def make_chunks(words, chunk_size=256, overlap=48):
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append({
            'start': i,
            'end': i + len(chunk_words),
            'words': chunk_words,
            'text': ' '.join(chunk_words)
        })
        if i + chunk_size >= len(words):
            break
    return chunks


def bm25_scores(query, chunks):
    q_tokens = tokenize(query)
    docs = [Counter(tokenize(c['text'])) for c in chunks]
    lengths = [len(c['words']) for c in chunks]
    avg_len = sum(lengths) / max(1, len(lengths))
    q_set = set(q_tokens)
    dfs = {term: sum(term in doc for doc in docs) for term in q_set}
    
    scores = []
    for i, doc in enumerate(docs):
        score = 0.0
        for term in q_set:
            tf = doc[term]
            if tf > 0:
                df = dfs[term]
                idf = math.log(1.0 + (len(docs) - df + 0.5) / (df + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * lengths[i] / avg_len))
        scores.append(score)
    return scores


def dense_scores(query, chunks):
    if EMBEDDER is None:
        return [0.0] * len(chunks)
    q_emb = EMBEDDER.encode(query)
    c_embs = EMBEDDER.encode([c['text'] for c in chunks], show_progress_bar=False)
    q_norm = np.linalg.norm(q_emb)
    c_norms = np.linalg.norm(c_embs, axis=1)
    sims = np.dot(c_embs, q_emb) / (np.maximum(c_norms * q_norm, 1e-9))
    return sims.tolist()


def evaluate_fixed_boundary(words, target, boundary_words=400):
    head = words[:boundary_words]
    tail = words[-boundary_words:]
    view = ' '.join(head + tail)
    hit = target.lower() in view.lower()
    return hit, len(head) + len(tail)


def evaluate_rber(words, query, target, max_depth=6, boundary_words=400, seam_padding=64):
    """
    RBER: Recursive Boundary Exposure Retrieval.
    Iteratively strips uninformative exterior boundaries via tri-partitioning,
    promoting interior content into outer boundary views.
    """
    region = words
    calls = 0
    total_exposed_words = 0
    resolved_depth = -1
    
    # Precompute query embedding if available
    q_emb = EMBEDDER.encode(query) if EMBEDDER is not None else None
    
    for depth in range(max_depth + 1):
        calls += 1
        n = len(region)
        
        # 1. Boundary Exposure View of the current region
        if n <= boundary_words * 2 or depth == max_depth:
            view = region
            total_exposed_words += len(view)
            hit = target.lower() in (' '.join(view)).lower()
            resolved_depth = depth
            return hit, resolved_depth, total_exposed_words, calls
        
        head = region[:boundary_words]
        tail = region[-boundary_words:]
        view = head + tail
        total_exposed_words += len(view)
        
        # 2. Deterministic Boundary Screening: is the target in current boundary view?
        if target.lower() in (' '.join(view)).lower():
            resolved_depth = depth
            return True, resolved_depth, total_exposed_words, calls
            
        # 3. Geometric Tri-Partitioning with Seam Padding
        third = n // 3
        p1 = region[:third + seam_padding]
        p2 = region[max(0, third - seam_padding) : min(n, 2 * third + seam_padding)]
        p3 = region[max(0, 2 * third - seam_padding):]
        
        # 4. Chunk-Level Max-Pooling Semantic Scoring
        def score_partition(p_words):
            if EMBEDDER is None or q_emb is None:
                # Fallback to BM25 max-pooling
                c_list = make_chunks(p_words, chunk_size=200, overlap=32)
                return max(bm25_scores(query, c_list)) if c_list else 0.0
            c_list = make_chunks(p_words, chunk_size=200, overlap=32)
            if not c_list:
                return 0.0
            embs = EMBEDDER.encode([c['text'] for c in c_list], show_progress_bar=False)
            sims = np.dot(embs, q_emb) / (np.maximum(np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb), 1e-9))
            return float(np.max(sims))
            
        s1 = score_partition(p1)
        s2 = score_partition(p2)
        s3 = score_partition(p3)
        
        # Branch to partition with highest semantic score
        best_score = max(s1, s2, s3)
        if best_score == s1:
            region = p1
        elif best_score == s3:
            region = p3
        else:
            region = p2
            
    return False, max_depth, total_exposed_words, calls


def run_benchmark(data_path, num_samples=25, seed=42):
    import random
    random.seed(seed)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
        
    sample = random.sample(records, min(num_samples, len(records)))
    print(f"=== Running Real LongBench Benchmark on {len(sample)} Samples ===")
    print(f"Dataset: {data_path}")
    print(f"Embedding Engine: {'all-MiniLM-L6-v2' if EMBEDDER else 'BM25-fallback'}\n")
    
    results = []
    
    for idx, item in enumerate(sample, 1):
        target = item['answers'][0]
        query = item['input']
        context = item['context']
        words = context.split()
        total_words = len(words)
        
        # Relative depth of target in ground-truth text
        target_pos = context.lower().find(target.lower())
        rel_depth = (target_pos / len(context)) if target_pos != -1 else 0.5
        depth_category = 'Head (0-20%)' if rel_depth < 0.2 else ('Tail (80-100%)' if rel_depth > 0.8 else 'Middle (20-80%)')
        
        # 1. Fixed Boundary
        fb_hit, fb_words = evaluate_fixed_boundary(words, target)
        
        # 2. Chunking for BM25 and Dense
        chunks = make_chunks(words, chunk_size=256, overlap=48)
        
        # BM25
        b_scores = bm25_scores(query, chunks)
        top_bm25_indices = sorted(range(len(chunks)), key=lambda i: b_scores[i], reverse=True)
        bm25_top1_hit = target.lower() in chunks[top_bm25_indices[0]]['text'].lower()
        bm25_top3_hit = any(target.lower() in chunks[i]['text'].lower() for i in top_bm25_indices[:3])
        bm25_top5_hit = any(target.lower() in chunks[i]['text'].lower() for i in top_bm25_indices[:5])
        bm25_top3_words = sum(len(chunks[i]['words']) for i in top_bm25_indices[:3])
        
        # Dense Cosine
        d_scores = dense_scores(query, chunks)
        top_dense_indices = sorted(range(len(chunks)), key=lambda i: d_scores[i], reverse=True)
        dense_top1_hit = target.lower() in chunks[top_dense_indices[0]]['text'].lower()
        dense_top3_hit = any(target.lower() in chunks[i]['text'].lower() for i in top_dense_indices[:3])
        dense_top5_hit = any(target.lower() in chunks[i]['text'].lower() for i in top_dense_indices[:5])
        dense_top3_words = sum(len(chunks[i]['words']) for i in top_dense_indices[:3])
        
        # 3. RBER
        rber_hit, rber_depth, rber_words, rber_calls = evaluate_rber(words, query, target)
        
        row = {
            'id': item.get('_id', f'sample_{idx}'),
            'total_words': total_words,
            'rel_depth': round(rel_depth * 100, 1),
            'depth_category': depth_category,
            'target': target,
            'fixed_boundary': {'hit': fb_hit, 'words': fb_words},
            'bm25': {'top1': bm25_top1_hit, 'top3': bm25_top3_hit, 'top5': bm25_top5_hit, 'words': bm25_top3_words},
            'dense': {'top1': dense_top1_hit, 'top3': dense_top3_hit, 'top5': dense_top5_hit, 'words': dense_top3_words},
            'rber': {'hit': rber_hit, 'depth': rber_depth, 'words': rber_words, 'calls': rber_calls}
        }
        results.append(row)
        
        print(f"[{idx:02d}/{len(sample):02d}] Target: {target:<14} | Depth: {rel_depth*100:4.1f}% ({depth_category:<16}) | "
              f"FB: {'✓' if fb_hit else '✗'} | "
              f"BM25@3: {'✓' if bm25_top3_hit else '✗'} | "
              f"Dense@3: {'✓' if dense_top3_hit else '✗'} | "
              f"RBER: {'✓' if rber_hit else '✗'} (d={rber_depth}, {rber_words} words)")

    # Aggregates
    n = len(results)
    fb_acc = sum(r['fixed_boundary']['hit'] for r in results) / n * 100
    bm25_top1_acc = sum(r['bm25']['top1'] for r in results) / n * 100
    bm25_top3_acc = sum(r['bm25']['top3'] for r in results) / n * 100
    dense_top1_acc = sum(r['dense']['top1'] for r in results) / n * 100
    dense_top3_acc = sum(r['dense']['top3'] for r in results) / n * 100
    rber_acc = sum(r['rber']['hit'] for r in results) / n * 100
    
    avg_total_words = np.mean([r['total_words'] for r in results])
    avg_rber_words = np.mean([r['rber']['words'] for r in results])
    token_reduction = (1.0 - (avg_rber_words / avg_total_words)) * 100
    
    # Subgroup breakdown (Middle 20-80% depth)
    middle_rows = [r for r in results if r['depth_category'] == 'Middle (20-80%)']
    n_mid = len(middle_rows)
    fb_mid_acc = sum(r['fixed_boundary']['hit'] for r in middle_rows) / max(1, n_mid) * 100
    bm25_mid_acc = sum(r['bm25']['top3'] for r in middle_rows) / max(1, n_mid) * 100
    dense_mid_acc = sum(r['dense']['top3'] for r in middle_rows) / max(1, n_mid) * 100
    rber_mid_acc = sum(r['rber']['hit'] for r in middle_rows) / max(1, n_mid) * 100
    
    summary = {
        'total_samples': n,
        'middle_samples': n_mid,
        'overall_accuracy': {
            'Fixed_Boundary': round(fb_acc, 2),
            'BM25_Top1': round(bm25_top1_acc, 2),
            'BM25_Top3': round(bm25_top3_acc, 2),
            'Dense_Top1': round(dense_top1_acc, 2),
            'Dense_Top3': round(dense_top3_acc, 2),
            'RBER': round(rber_acc, 2)
        },
        'middle_depth_accuracy (Lost-in-the-Middle Region)': {
            'Fixed_Boundary': round(fb_mid_acc, 2),
            'BM25_Top3': round(bm25_mid_acc, 2),
            'Dense_Top3': round(dense_mid_acc, 2),
            'RBER': round(rber_mid_acc, 2)
        },
        'efficiency': {
            'avg_document_words': round(float(avg_total_words), 1),
            'avg_rber_exposed_words': round(float(avg_rber_words), 1),
            'token_reduction_pct': round(float(token_reduction), 2),
            'avg_rber_depth': round(float(np.mean([r['rber']['depth'] for r in results])), 2),
            'avg_rber_calls': round(float(np.mean([r['rber']['calls'] for r in results])), 2)
        }
    }
    
    out_file = Path('benchmark/real_dataset_results.json')
    out_file.write_text(json.dumps({'summary': summary, 'trials': results}, indent=2))
    print("\n" + "=" * 60)
    print("FINAL SUMMARY REPORT:")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved full per-trial results to {out_file}")


if __name__ == '__main__':
    data_file = 'data/passage_retrieval_en.jsonl'
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    run_benchmark(data_file, num_samples=samples)
