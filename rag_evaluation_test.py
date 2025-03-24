import json
import os
from datetime import datetime
from typing import List, Dict

class TestCaseManager:
    """Manages the creation and handling of test cases for human history data."""

    def __init__(self, json_file="rag_evaluation_test_cases.json"):
        self.json_file = json_file
        self.test_cases = self.load_test_cases()
        if not self.test_cases:  # Initialize with predefined test cases if file is empty
            self.initialize_test_cases()

    def load_test_cases(self) -> List[Dict]:
        """Loads test cases from a JSON file."""
        if os.path.exists(self.json_file):
            with open(self.json_file, "r", encoding="utf-8") as file:
                return json.load(file)
        return []

    def save_test_cases(self) -> None:
        """Saves test cases to a JSON file."""
        with open(self.json_file, "w", encoding="utf-8") as file:
            json.dump(self.test_cases, file, indent=4)

    def add_test_case(self, test_case: Dict) -> None:
        """Adds a new test case to the collection and saves it."""
        self.test_cases.append(test_case)
        self.save_test_cases()

    def get_test_case(self, test_case_id: str) -> Dict:
        """Retrieves a test case by ID."""
        for test in self.test_cases:
            if test["TEST CASE ID"] == test_case_id:
                return test
        return {}

    def update_test_case(self, test_case_id: str, update_data: Dict) -> bool:
        """Updates a test case with new information."""
        for test in self.test_cases:
            if test["TEST CASE ID"] == test_case_id:
                test.update(update_data)
                self.save_test_cases()
                return True
        return False

    def log_test_result(self, test_case_id: str, actual_result: str, status: str) -> None:
        """Logs the actual result and status of a test case."""
        self.update_test_case(test_case_id, {
            "ACTUAL RESULT": actual_result,
            "STATUS": status,
            "TIMESTAMP": datetime.now().isoformat()
        })

    def initialize_test_cases(self) -> None:
        """Initializes test cases with predefined data and saves them."""
        self.test_cases = [
            {
                "TEST CASE ID": "TC001",
                "SECTION": "Human History",
                "SUB-SECTION": "Sexual Division of Labor",
                "TEST CASE TITLE": "Understanding SDL in Hunter-Gatherer Societies",
                "TEST DESCRIPTION": "Evaluate how the sexual division of labor differs among human hunter-gatherer societies compared to other species.",
                "PRECONDITIONS": "Knowledge of hunter-gatherer lifestyles and food acquisition patterns.",
                "TEST DATA": {
                    "context": "Sexual division of labour (SDL) is the delegation of different tasks between the male and female members of a species...",
                    "question": "What is the sexual division of labor (SDL) and how does it differ among human hunter-gatherer societies compared to other species?",
                    "expected_answer": "The Sexual Division of Labor (SDL) refers to the delegation of different tasks between male and female members of a species..."
                },
                "TEST STEPS": [
                    "Read the provided context.",
                    "Analyze the gender-based division of labor in hunter-gatherer societies.",
                    "Compare SDL in humans with other species."
                ],
                "EXPECTED RESULT": "A clear understanding of the combined food acquisition and sharing model unique to humans.",
                "ACTUAL RESULT": "",
                "STATUS": ""
            },
            {
                "TEST CASE ID": "TC002",
                "SECTION": "Human History",
                "SUB-SECTION": "Gendered Roles in Foraging Societies",
                "TEST CASE TITLE": "Historical Perspective on Gendered Roles",
                "TEST DESCRIPTION": "Evaluate the historical perspective on gendered roles in hunter-gatherer societies and identify key anthropologists.",
                "PRECONDITIONS": "Familiarity with anthropological studies on early human societies.",
                "TEST DATA": {
                    "context": "From the 1970s onward, the dominant paleontological perspective of gendered roles in hunter-gatherer societies was of a model...",
                    "question": "What is the historical perspective on gendered roles in hunter-gatherer societies, as highlighted in the text, and who were the anthropologists who coined this perspective?",
                    "expected_answer": "The historical perspective on gendered roles in hunter-gatherer societies was primarily a model termed 'Man the Hunter, Woman the Gatherer'..."
                },
                "TEST STEPS": [
                    "Review the historical model.",
                    "Identify anthropologists who proposed the theory.",
                    "Compare it with modern research findings."
                ],
                "EXPECTED RESULT": "Understanding of the 'Man the Hunter, Woman the Gatherer' model and its critiques.",
                "ACTUAL RESULT": "",
                "STATUS": ""
            }
        ]
        self.save_test_cases()

# Example Usage
test_manager = TestCaseManager()
test_cases = test_manager.load_test_cases()
print("Loaded Test Cases:", test_cases)
