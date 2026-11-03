"""CPU-only, oracle exposure benchmark; no Ollama calls or generated-answer claims."""
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Importing the backend initializes its DB. Keep the benchmark isolated.
TEMP = tempfile.TemporaryDirectory()
os.environ['RBER_DATA_DIR'] = TEMP.name
from backend.app import boundary_view, make_chunks, MIN_REGION_WORDS, MAX_DEPTH

NOTE = 'Note: SECRET KEY: The combination to the safe is "AGI is Coming".'
NEEDLE = 'AGI is Coming'
QUERIES = {'lexical': 'secret key combination safe', 'paraphrase': 'what phrase unlocks the vault'}


def tokenize(text):
    return re.findall(r'\w+', text.lower())


def bm25_top(words, query):
    chunks, _ = make_chunks([(None, ' '.join(words))])
    docs = [Counter(tokenize(c['text'])) for c in chunks]
    lengths = [sum(c.values()) for c in docs]
    avg = sum(lengths) / len(lengths)
    query_terms = set(tokenize(query))
    dfs = {q: sum(q in doc for doc in docs) for q in query_terms}
    def score(i):
        total = 0.0
        for term in query_terms:
            tf = docs[i][term]
            if tf:
                idf = math.log(1 + (len(docs) - dfs[term] + 0.5) / (dfs[term] + 0.5))
                total += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * lengths[i] / avg))
        return total
    best = max(range(len(docs)), key=score)
    return NEEDLE in chunks[best]['text'], len(chunks[best]['text'].split())


def rber_oracle(words):
    region = words
    calls = 0
    exposed_words = 0
    for depth in range(MAX_DEPTH + 1):
        if len(region) <= MIN_REGION_WORDS or depth == MAX_DEPTH:
            regions = [region]
            middle = None
        else:
            third = len(region) // 3
            regions = [region[:third], region[2 * third:]]
            middle = region[third:2 * third]
        for region_part in regions:
            view = boundary_view(region_part)
            calls += 1
            exposed_words += len(view)
            if NEEDLE in ' '.join(view):
                return True, calls, exposed_words
        if middle is None:
            break
        region = middle
    return False, calls, exposed_words


def main():
    original = (ROOT / 'long.txt').read_text(encoding='utf-8')
    if original.count(NOTE) != 1:
        raise SystemExit('Expected exactly one original note in long.txt')
    background = original.replace(NOTE, '').split()
    output = []
    for query_kind, query in QUERIES.items():
        for step in range(21):
            idx = round(step * len(background) / 20)
            words = background[:idx] + NOTE.split() + background[idx:]
            hit, calls, exposure = rber_oracle(words)
            bm_hit, bm_words = bm25_top(words, query)
            fixed = boundary_view(words)
            output.append({'query': query_kind, 'position_percent': step * 5,
                           'rber_hit': hit, 'rber_calls': calls, 'rber_words': exposure,
                           'bm25_hit': bm_hit, 'bm25_words': bm_words,
                           'fixed_hit': NEEDLE in ' '.join(fixed), 'fixed_words': len(fixed),
                           'full_hit': NEEDLE in ' '.join(words), 'full_words': len(words)})
    summary = {}
    for kind in QUERIES:
        rows = [r for r in output if r['query'] == kind]
        summary[kind] = {
            'trials': len(rows),
            'rber_oracle_hits': sum(r['rber_hit'] for r in rows),
            'rber_avg_calls': round(statistics.mean(r['rber_calls'] for r in rows), 2),
            'rber_avg_words': round(statistics.mean(r['rber_words'] for r in rows)),
            'bm25_hits': sum(r['bm25_hit'] for r in rows),
            'bm25_avg_words': round(statistics.mean(r['bm25_words'] for r in rows)),
            'fixed_hits': sum(r['fixed_hit'] for r in rows),
            'fixed_avg_words': round(statistics.mean(r['fixed_words'] for r in rows)),
            'full_hits': sum(r['full_hit'] for r in rows),
            'full_avg_words': round(statistics.mean(r['full_words'] for r in rows)),
            'rber_missed_positions': [r['position_percent'] for r in rows if not r['rber_hit']],
        }
    print(json.dumps({'summary': summary, 'trials': output}, indent=2))


if __name__ == '__main__':
    main()
