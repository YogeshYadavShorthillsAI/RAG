import unittest
import json
import os
from datetime import datetime
from unittest.mock import patch, MagicMock
from evaluation import process_test_cases

from evaluation import (
    calculate_metrics, 
    qa_pipeline, 
    log_interaction, 
    load_test_cases, 
    process_test_cases,
    calculate_grade,
    LOG_FILE, RESULTS_FILE, SUMMARY_FILE
)

class TestQAPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_question = "Who was the first president of the United States?"
        cls.test_context = "George Washington was the first president of the United States, serving from 1789 to 1797."
        cls.test_answer = "George Washington."
    
    def test_calculate_metrics(self):
        generated = "George Washington was the first U.S. president."
        reference = "George Washington."
        metrics = calculate_metrics(generated, reference)
        self.assertIn("rouge_score", metrics)
        self.assertIn("cosine_similarity", metrics)
        self.assertIn("bert_score_f1", metrics)
        self.assertIn("final_score", metrics)
        self.assertTrue(0 <= metrics["final_score"] <= 1)
    
    @patch("requests.post")
    def test_qa_pipeline(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "George Washington."}}]}
        mock_post.return_value = mock_response
        response = qa_pipeline(self.test_question, self.test_context)
        self.assertEqual(response, "George Washington.")
    
    def test_log_interaction(self):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": self.test_question,
            "context": self.test_context,
            "generated_answer": "George Washington.",
            "reference_answer": self.test_answer,
            "rouge_score": 1.0,
            "cosine_similarity": 1.0,
            "bert_score_f1": 1.0,
            "final_score": 1.0
        }
        log_interaction(entry)
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
            self.assertTrue(any(log["question"] == self.test_question for log in logs))
    
    def test_calculate_grade(self):
        self.assertEqual(calculate_grade(0.95), "A (Excellent)")
        self.assertEqual(calculate_grade(0.85), "B (Good)")
        self.assertEqual(calculate_grade(0.75), "C (Average)")
        self.assertEqual(calculate_grade(0.65), "D (Below Average)")
        self.assertEqual(calculate_grade(0.50), "F (Poor)")
    
    @patch("evaluation.load_test_cases", return_value=[
        {"question": "Who was the first president?", "context": "George Washington was the first president.", "answer": "George Washington."}
    ])
    @patch("evaluation.qa_pipeline", return_value="George Washington.")
    def test_process_test_cases(self, mock_qa, mock_test_cases):
        results = process_test_cases()
        self.assertTrue(os.path.exists(RESULTS_FILE))
        with open(RESULTS_FILE, "r") as f:
            results_data = json.load(f)
            self.assertGreater(len(results_data), 0)
        self.assertTrue(os.path.exists(SUMMARY_FILE))
        with open(SUMMARY_FILE, "r") as f:
            summary_data = json.load(f)
            self.assertIn("grade", summary_data)
    
if __name__ == "__main__":
    unittest.main()
