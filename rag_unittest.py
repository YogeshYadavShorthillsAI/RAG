import unittest
import os
import json
from Rag_pipeline import get_vectorizer
import shutil

class TestHumanHistoryVectorizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup before all tests."""
        cls.vectorizer = get_vectorizer()

    def test_text_file_reading(self):
        """Test if the text file reading works correctly."""
        test_text = "This is a test string."
        with open("test_text.txt", "w") as f:
            f.write(test_text)

        self.vectorizer.text_file = "test_text.txt"
        result = self.vectorizer.read_text_file()
        os.remove("test_text.txt")

        self.assertEqual(result, test_text, "Text file reading failed!")

    def test_chunk_creation(self):
        """Test text chunking."""
        test_text = "This is a test string. " * 50  # Large enough for chunking
        chunks = self.vectorizer.create_chunks(test_text)
        
        self.assertGreater(len(chunks), 0, "Chunking failed - No chunks created.")

    def test_faiss_index_creation(self):
        """Test FAISS index creation."""
        test_text = "This is a test string. " * 50
        chunks = self.vectorizer.create_chunks(test_text)
        self.vectorizer.create_faiss_index(chunks)
        
        self.assertTrue(os.path.exists(self.vectorizer.faiss_index_path), "FAISS index was not created!")

    def test_query_logging(self):
        """Test if query logging works."""
        log_file = "query_logs.json"

        # Ensure log file is cleared
        if os.path.exists(log_file):
            os.remove(log_file)

        self.vectorizer._log_query("Test question?", "Test answer.")

        # Read log file and check content
        with open(log_file, "r") as f:
            logs = json.load(f)

        self.assertEqual(len(logs), 1, "Query log entry was not created.")
        self.assertEqual(logs[0]["question"], "Test question?", "Logged question is incorrect.")
        self.assertEqual(logs[0]["answer"], "Test answer.", "Logged answer is incorrect.")
    @classmethod
    def tearDownClass(cls):
        """Cleanup after all tests."""
        if os.path.exists(cls.vectorizer.faiss_index_path):
            if os.path.isfile(cls.vectorizer.faiss_index_path):
                os.remove(cls.vectorizer.faiss_index_path)  # If it's a file
            elif os.path.isdir(cls.vectorizer.faiss_index_path):
                shutil.rmtree(cls.vectorizer.faiss_index_path)  # If it's a directory

if __name__ == "__main__":
    unittest.main()
