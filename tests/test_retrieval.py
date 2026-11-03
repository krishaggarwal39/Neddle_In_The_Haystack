import json
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app import (boundary_view, make_chunks, passage_parts, rank_chunks,
                         reconstruct_words, search_boundaries)


class RetrievalTests(unittest.TestCase):
    def test_overlapping_index_reconstructs_document(self):
        text = ' '.join(f'word{i}' for i in range(1000))
        chunks, count = make_chunks([(None, text)])
        self.assertEqual(count, 1000)
        self.assertEqual(reconstruct_words(chunks), text.split())

    def test_character_split_passages_reconstruct_document(self):
        text = ' '.join(f'word{i}' for i in range(900))
        chunks, _ = make_chunks([(None, text)])
        split = [{'text': part} for chunk in chunks for part in passage_parts(chunk['text'])]
        self.assertEqual(reconstruct_words(split), text.split())

    def test_boundary_view_exposes_start_and_end(self):
        words = list(map(str, range(2000)))
        view = boundary_view(words)
        self.assertEqual(view, words[:400] + words[-400:])

    def test_evaluates_left_then_right_and_peels_middle(self):
        words = [f'w{i}' for i in range(2700)]
        chunks, _ = make_chunks([(None, ' '.join(words))])
        for idx, chunk in enumerate(chunks):
            chunk.update(document_id='x', idx=idx, score=1)
        views = []
        def answer(question, view):
            views.append(view)
            return 'found' if len(views) == 3 else None
        with patch('backend.app.evaluate_boundary', side_effect=answer):
            result, sources, trace = search_boundaries('question', chunks, {'x': 'doc'})
        self.assertEqual(result, 'found')
        self.assertEqual(views[0], words[:400] + words[500:900])
        self.assertEqual(views[1], words[1800:2200] + words[2300:2700])
        self.assertEqual(views[2], words[900:1200])
        self.assertEqual(len(trace[0]['steps']), 2)
        self.assertTrue(sources[0]['boundary_view'])

    def test_long_document_exposes_safe_secret(self):
        text = (Path(__file__).resolve().parent.parent / 'long.txt').read_text()
        chunks, _ = make_chunks([(None, text)])
        for idx, chunk in enumerate(chunks):
            chunk.update(document_id='x', idx=idx, score=0.5)
        quote = 'The combination to the safe is "AGI is Coming".'
        def answer(question, view):
            return quote if quote in ' '.join(view) else None
        with patch('backend.app.evaluate_boundary', side_effect=answer) as evaluate:
            result, sources, trace = search_boundaries('What is the secret to the safe?', chunks, {'x': 'long.txt'})
        self.assertEqual(result, quote)
        self.assertIn(quote, sources[0]['text'])
        self.assertLessEqual(evaluate.call_count, 6)

    def test_not_found_after_examining_core(self):
        chunks, _ = make_chunks([(None, ' '.join(f'word{i}' for i in range(2000)))])
        for idx, chunk in enumerate(chunks):
            chunk.update(document_id='x', idx=idx, score=0.5)
        with patch('backend.app.evaluate_boundary', return_value=None) as evaluate:
            result, sources, trace = search_boundaries('question', chunks, {'x': 'doc'})
        self.assertIsNone(result)
        self.assertEqual(sources, [])
        self.assertEqual(evaluate.call_count, 5)

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            rank_chunks([{'text': 'x', 'embedding': json.dumps([1, 2])}], [1], 'x')


if __name__ == '__main__':
    unittest.main()
