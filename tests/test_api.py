import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import backend.app as app_module

class ApiTests(unittest.TestCase):
    def test_upload_query_delete(self):
        # Use an isolated database and deterministic stand-ins for Ollama.
        with tempfile.TemporaryDirectory() as temp, patch.object(app_module, 'DB', Path(temp) / 'test.sqlite3'):
            with app_module.db() as conn:
                conn.executescript('''CREATE TABLE documents (id TEXT PRIMARY KEY, name TEXT, status TEXT, error TEXT, pages INTEGER DEFAULT 0, words INTEGER DEFAULT 0, chunks INTEGER DEFAULT 0, embedding_model TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP); CREATE TABLE chunks (document_id TEXT, idx INTEGER, page INTEGER, start_word INTEGER, end_word INTEGER, text TEXT, embedding TEXT, PRIMARY KEY(document_id, idx));''')
            with patch.object(app_module, 'embed', side_effect=lambda texts, query=False: [[1.0, 0.0] for _ in texts]), patch.object(app_module, 'ollama_post', return_value={'message': {'content': 'The safe code is 2004.'}}):
                client = TestClient(app_module.app)
                response = client.post('/api/documents', files={'file': ('note.txt', b'The safe code is 2004.', 'text/plain')})
                self.assertEqual(response.status_code, 202)
                doc_id = response.json()['id']
                self.assertEqual(client.get('/api/documents').json()[0]['status'], 'ready')
                response = client.post('/api/ask', json={'question': 'What is the safe code?', 'document_ids': [doc_id]})
                self.assertEqual(response.status_code, 200)
                self.assertIn('The safe code is 2004.', response.json()['answer'])
                self.assertEqual(response.json()['sources'][0]['document'], 'note.txt')
                self.assertEqual(client.delete('/api/documents/' + doc_id).status_code, 200)
                self.assertEqual(client.get('/api/documents').json(), [])

    def test_rejects_multiple_questions_and_busy_requests(self):
        client = TestClient(app_module.app)
        payload = {'question': 'First? Second?', 'document_ids': ['x']}
        self.assertEqual(client.post('/api/ask', json=payload).status_code, 400)
        with app_module.ask_lock:
            response = client.post('/api/ask', json={'question': 'One?', 'document_ids': ['x']})
        self.assertEqual(response.status_code, 429)
        self.assertFalse(app_module.ask_lock.locked())

if __name__ == '__main__': unittest.main()
