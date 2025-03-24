import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import json
import numpy as np
from datetime import datetime
from evaluation import (  # Replace with your actual module name
    setup_logger,
    log_interaction,
    calculate_metrics,
    load_test_cases,
    qa_pipeline,
    process_test_cases,
    calculate_grade,
    METRIC_WEIGHTS
)

class TestReporter:
    """Handles test reporting and saving results to JSON."""
    
    def __init__(self, report_file="evaluation_test_report.json"):
        self.report_file = report_file
        self.test_cases = []
        self.current_test_id = 1
    
    def add_test_case(
        self,
        section: str,
        subsection: str,
        title: str,
        description: str,
        preconditions: str,
        test_data: str,
        steps: list,
        expected_result: str,
        actual_result: str,
        status: str
    ) -> dict:
        """Adds a test case to the report."""
        test_case = {
            "TEST CASE ID": f"TC{self.current_test_id:03d}",
            "SECTION": section,
            "SUB-SECTION": subsection,
            "TEST CASE TITLE": title,
            "TEST DESCRIPTION": description,
            "PRECONDITIONS": preconditions,
            "TEST DATA": test_data,
            "TEST STEPS": steps,
            "EXPECTED RESULT": expected_result,
            "ACTUAL RESULT": actual_result,
            "STATUS": status,
            "TIMESTAMP": datetime.now().isoformat()
        }
        self.test_cases.append(test_case)
        self.current_test_id += 1
        return test_case
    
    def save_report(self) -> None:
        """Saves the test report to JSON file."""
        with open(self.report_file, 'w') as f:
            json.dump({"test_cases": self.test_cases}, f, indent=4)

class TestQAEvaluationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reporter = TestReporter()
    
    @classmethod
    def tearDownClass(cls):
        cls.reporter.save_report()
        print(f"\nTest Report generated: {cls.reporter.report_file}")
        print(f"Total Tests: {cls.reporter.current_test_id - 1}")

    def setUp(self):
        # Sample test data
        self.sample_test_cases = [
            {
                "question": "What is the capital of France?",
                "context": "France is a country in Europe. Its capital is Paris.",
                "answer": "Paris"
            }
        ]
        self.sample_results = {
            "rouge_score": 0.95,
            "cosine_similarity": 0.92,
            "bert_score_f1": 0.93,
            "final_score": 0.93
        }
        self.sample_log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": "What is the capital of France?",
            "context": "France is a country in Europe. Its capital is Paris.",
            "generated_answer": "Paris",
            "reference_answer": "Paris",
            **self.sample_results
        }

    def test_setup_logger_creates_file(self):
        test_case = self.reporter.add_test_case(
            section="Logger",
            subsection="Setup",
            title="Logger Creates File",
            description="Test that logger creates file when none exists",
            preconditions="Log file does not exist",
            test_data="None",
            steps=[
                "Call setup_logger() when file doesn't exist",
                "Verify file is created with empty array"
            ],
            expected_result="File should be created with empty JSON array",
            actual_result="",
            status="RUNNING"
        )
        
        try:
            with patch("builtins.open", mock_open()) as mock_file:
                with patch("os.path.exists", return_value=False):
                    setup_logger()
                    mock_file.assert_called_once_with("qa_interactions.log", "w")
                    mock_file().write.assert_called_once_with("[]")
                    
                    test_case["ACTUAL RESULT"] = "File created with empty array"
                    test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

    @patch("os.path.exists", return_value=True)
    @patch("os.stat")
    def test_setup_logger_empty_file(self, mock_stat, mock_exists):
        test_case = self.reporter.add_test_case(
            section="Logger",
            subsection="Setup",
            title="Logger Handles Empty File",
            description="Test that logger handles empty existing file",
            preconditions="Log file exists but is empty",
            test_data="Empty file",
            steps=[
                "Call setup_logger() with empty file",
                "Verify file is initialized with empty array"
            ],
            expected_result="File should be initialized with empty JSON array",
            actual_result="",
            status="RUNNING"
        )
        
        try:
            mock_stat.return_value.st_size = 0
            with patch("builtins.open", mock_open()) as mock_file:
                setup_logger()
                mock_file.assert_called_once_with("qa_interactions.log", "w")
                mock_file().write.assert_called_once_with("[]")
                
                test_case["ACTUAL RESULT"] = "File initialized with empty array"
                test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps([{"existing": "entry"}]))
    @patch("os.path.exists", return_value=True)
    @patch("os.stat")
    def test_log_interaction(self, mock_stat, mock_exists, mock_file):
        test_case = self.reporter.add_test_case(
            section="Logger",
            subsection="Interaction",
            title="Log Interaction",
            description="Test logging a new interaction",
            preconditions="Log file exists with content",
            test_data="Sample log entry",
            steps=[
                "Call log_interaction() with new entry",
                "Verify entry is appended to log"
            ],
            expected_result="New entry should be added to log file",
            actual_result="",
            status="RUNNING"
        )
        
        try:
            mock_stat.return_value.st_size = 100
            log_interaction(self.sample_log_entry)
            mock_file.assert_called_with("qa_interactions.log", "w")
            self.assertTrue(mock_file().write.called)
            
            test_case["ACTUAL RESULT"] = "Entry successfully added to log"
            test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

    @patch("builtins.open", side_effect=json.JSONDecodeError("Error", "doc", 0))
    def test_log_interaction_corrupted_file(self, mock_file):
        test_case = self.reporter.add_test_case(
            section="Logger",
            subsection="Error Handling",
            title="Log Interaction Corrupted File",
            description="Test handling of corrupted log file",
            preconditions="Log file exists but is corrupted",
            test_data="Corrupted JSON file",
            steps=[
                "Call log_interaction() with corrupted file",
                "Verify new file is created with the entry"
            ],
            expected_result="Should create new file with the entry",
            actual_result="",
            status="RUNNING"
        )
        
        try:
            with patch("os.path.exists", return_value=True):
                with patch("os.stat"):
                    log_interaction(self.sample_log_entry)
                    mock_file.assert_called_with("qa_interactions.log", "w")
                    
                    test_case["ACTUAL RESULT"] = "New file created with entry"
                    test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

    def test_calculate_metrics(self):
        test_case = self.reporter.add_test_case(
            section="Metrics",
            subsection="Calculation",
            title="Calculate Metrics",
            description="Test metric calculation logic",
            preconditions="Sample generated and reference text",
            test_data="Generated: 'test', Reference: 'test'",
            steps=[
                "Call calculate_metrics() with test inputs",
                "Verify all metrics are calculated correctly"
            ],
            expected_result="Should return correct metric scores",
            actual_result="",
            status="RUNNING"
        )
        
        try:
            with patch("numpy.linalg.norm", return_value=1.0), \
                 patch("numpy.dot", return_value=0.9), \
                 patch("bert_score.score") as mock_bert:
                
                mock_bert.return_value = (MagicMock(), MagicMock(), MagicMock())
                mock_bert.return_value[2].mean.return_value = 0.85
                
                with patch("rouge_score.rouge_scorer.RougeScorer.score") as mock_rouge:
                    mock_rouge.return_value = {"rougeL": MagicMock(fmeasure=0.8)}
                    
                    metrics = calculate_metrics("test", "test")
                    
                    expected_final = (
                        0.25 * 0.8 +  # rouge
                        0.25 * 0.9 +  # cosine
                        0.5 * 0.85    # bert
                    )
                    
                    self.assertAlmostEqual(metrics["final_score"], expected_final)
                    self.assertEqual(metrics["rouge_score"], 0.8)
                    self.assertEqual(metrics["cosine_similarity"], 0.9)
                    self.assertEqual(metrics["bert_score_f1"], 0.85)
                    
                    test_case["ACTUAL RESULT"] = f"Metrics calculated correctly: {metrics}"
                    test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

    # [Additional test methods with similar structure...]

if __name__ == "__main__":
    unittest.main()