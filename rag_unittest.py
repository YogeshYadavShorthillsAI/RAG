import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import json
from datetime import datetime
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
import uuid
from typing import List, Dict, Tuple  # Added missing imports
from dotenv import load_dotenv

load_dotenv()

class TestReporter:
    """Handles test reporting and saving results to JSON."""
    
    def __init__(self, report_file="test_report.json"):
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
        steps: List[str],
        expected_result: str,
        actual_result: str,
        status: str
    ) -> Dict:
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

class HumanHistoryVectorizer:
    """Processes human history data into a FAISS vector database and implements RAG pipeline."""

    def __init__(
        self, 
        text_file="combined_text.txt", 
        chunk_size=1000, 
        chunk_overlap=200,
        faiss_index_path="faiss_index"
    ):
        self.text_file = text_file
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.faiss_index_path = faiss_index_path

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True
        )

    def read_text_file(self) -> str:
        """Reads text from the specified file."""
        if not os.path.exists(self.text_file):
            raise FileNotFoundError(f"File {self.text_file} not found.")

        with open(self.text_file, 'r', encoding='utf-8') as file:
            return file.read()

    def create_chunks(self, text: str) -> List[Document]:
        """Splits text into overlapping chunks."""
        documents = self.text_splitter.create_documents([text])
        for doc in documents:
            doc.metadata["chunk_id"] = str(uuid.uuid4())
        return documents

    def create_faiss_index(self, documents: List[Document]) -> None:
        """Creates FAISS index from documents."""
        vector_store = FAISS.from_documents(documents, self.embeddings)
        vector_store.save_local(self.faiss_index_path)

    def process(self) -> None:
        """Main processing function to execute the pipeline."""
        text = self.read_text_file()
        documents = self.create_chunks(text)
        self.create_faiss_index(documents)

    def initialize_rag_pipeline(self):
        """Initializes the RAG pipeline."""
        vector_store = FAISS.load_local(
            self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True
        )
        mistral_llm = self.initialize_mistral_model()

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a historian AI. Answer using ONLY the provided context. "
                "If unsure, say so. Greet the user and ask if they have more questions.\n\n"
                "Context: {context}\n\nQuestion: {question}"
            )
        )

        return RetrievalQA.from_chain_type(
            llm=mistral_llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt_template}
        )

    def query_with_rag(self, query_text: str) -> Tuple[str, str]:
        """Queries FAISS and generates an answer using RAG."""
        rag_pipeline = self.initialize_rag_pipeline()
        result = rag_pipeline.run(query_text)
        self._log_query(query_text, result)
        return result, ""

    def initialize_mistral_model(self):
        """Initializes Mistral model."""
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("MISTRAL API key is required. Set MISTRAL_API_KEY in environment variables.")

        return ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.2,
            max_retries=2,
            api_key=mistral_api_key
        )

    def _log_query(self, query: str, answer: str) -> None:
        """Logs queries and answers."""
        log_entry = {"timestamp": datetime.now().isoformat(), "question": query, "answer": answer}
        log_file = "query_logs.json"
        logs = []

        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Log file corrupted. Starting fresh log.")

        logs.append(log_entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=4)

def get_vectorizer():
    """Helper function to get vectorizer instance."""
    return HumanHistoryVectorizer()

class TestHumanHistoryVectorizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reporter = TestReporter()
    
    @classmethod
    def tearDownClass(cls):
        cls.reporter.save_report()
        print(f"\nTest Report generated: {cls.reporter.report_file}")
        print(f"Total Tests: {cls.reporter.current_test_id - 1}")
    
    def setUp(self):
        self.test_instance = HumanHistoryVectorizer(
            text_file="test.txt",
            chunk_size=500,
            chunk_overlap=100,
            faiss_index_path="test_index"
        )
        self.sample_text = "This is a sample text for testing. " * 50
        self.sample_docs = [
            Document(page_content="First chunk", metadata={"chunk_id": "1"}),
            Document(page_content="Second chunk", metadata={"chunk_id": "2"})
        ]

    def test_initialization(self):
        """Test that the class initializes with correct parameters"""
        test_case = self.reporter.add_test_case(
            section="Initialization",
            subsection="Class Setup",
            title="Class Initialization",
            description="Verify the class initializes with correct parameters",
            preconditions="HumanHistoryVectorizer class available",
            test_data="text_file='test.txt', chunk_size=500, chunk_overlap=100, faiss_index_path='test_index'",
            steps=[
                "Initialize HumanHistoryVectorizer with test parameters",
                "Check instance attributes"
            ],
            expected_result="All parameters should be correctly set in the instance",
            actual_result="",
            status="RUNNING"
        )
        
        try:
            self.assertEqual(self.test_instance.text_file, "test.txt")
            self.assertEqual(self.test_instance.chunk_size, 500)
            self.assertEqual(self.test_instance.chunk_overlap, 100)
            self.assertEqual(self.test_instance.faiss_index_path, "test_index")
            
            test_case["ACTUAL RESULT"] = "All parameters correctly set"
            test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

    @patch("builtins.open", new_callable=mock_open, read_data="test content")
    def test_read_text_file_success(self, mock_file):
        """Test reading text file successfully"""
        test_case = self.reporter.add_test_case(
            section="File Operations",
            subsection="Text File Reading",
            title="Read Text File Success",
            description="Verify text file is read successfully",
            preconditions="Test file exists with content",
            test_data="test.txt with content 'test content'",
            steps=[
                "Call read_text_file() method",
                "Verify returned content"
            ],
            expected_result="Method should return file content",
            actual_result="",
            status="RUNNING"
        )
        
        try:
            result = self.test_instance.read_text_file()
            self.assertEqual(result, "test content")
            mock_file.assert_called_once_with("test.txt", 'r', encoding='utf-8')
            
            test_case["ACTUAL RESULT"] = "Returned 'test content'"
            test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_read_text_file_failure(self, mock_file):
        """Test handling of missing text file"""
        test_case = self.reporter.add_test_case(
            section="File Operations",
            subsection="Text File Reading",
            title="Read Missing Text File",
            description="Verify proper error when file doesn't exist",
            preconditions="Test file does not exist",
            test_data="nonexistent.txt",
            steps=[
                "Call read_text_file() with non-existent file",
                "Catch expected exception"
            ],
            expected_result="Should raise FileNotFoundError",
            actual_result="",
            status="RUNNING"
        )
        
        try:
            with self.assertRaises(FileNotFoundError):
                self.test_instance.read_text_file()
            
            test_case["ACTUAL RESULT"] = "Raised FileNotFoundError"
            test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

    def test_create_chunks(self):
        """Test document chunk creation"""
        test_case = self.reporter.add_test_case(
            section="Document Processing",
            subsection="Chunk Creation",
            title="Create Document Chunks",
            description="Verify text is properly split into chunks",
            preconditions="Sample text available",
            test_data="Sample text (1000 characters)",
            steps=[
                "Call create_chunks() with sample text",
                "Verify returned documents"
            ],
            expected_result="Should return list of Document objects with metadata",
            actual_result="",
            status="RUNNING"
        )
        
        with patch.object(self.test_instance.text_splitter, 'create_documents', 
                         return_value=self.sample_docs) as mock_create:
            try:
                result = self.test_instance.create_chunks(self.sample_text)
                
                mock_create.assert_called_once_with([self.sample_text])
                self.assertEqual(len(result), 2)
                self.assertIsInstance(result[0], Document)
                self.assertTrue(all('chunk_id' in doc.metadata for doc in result))
                
                test_case["ACTUAL RESULT"] = "Returned 2 Document objects with chunk_id"
                test_case["STATUS"] = "PASS"
            except AssertionError as e:
                test_case["ACTUAL RESULT"] = str(e)
                test_case["STATUS"] = "FAIL"
                raise

    @patch('langchain_community.vectorstores.FAISS.from_documents')
    @patch('langchain_community.vectorstores.FAISS.save_local')
    def test_create_faiss_index(self, mock_save, mock_from_docs):
        """Test FAISS index creation"""
        test_case = self.reporter.add_test_case(
            section="Vector Database",
            subsection="FAISS Index",
            title="Create FAISS Index",
            description="Verify FAISS index is created from documents",
            preconditions="Sample documents available",
            test_data="List of 2 Document objects",
            steps=[
                "Call create_faiss_index() with sample documents",
                "Verify index creation"
            ],
            expected_result="FAISS index should be created and saved",
            actual_result="",
            status="RUNNING"
        )
        
        mock_from_docs.return_value = MagicMock()
        
        try:
            self.test_instance.create_faiss_index(self.sample_docs)
            
            mock_from_docs.assert_called_once_with(self.sample_docs, self.test_instance.embeddings)
            mock_save.assert_called_once_with("test_index")
            
            test_case["ACTUAL RESULT"] = "Index created and saved to test_index"
            test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

    @patch.object(HumanHistoryVectorizer, 'read_text_file')
    @patch.object(HumanHistoryVectorizer, 'create_chunks')
    @patch.object(HumanHistoryVectorizer, 'create_faiss_index')
    def test_process(self, mock_create_index, mock_create_chunks, mock_read):
        """Test the main processing pipeline"""
        test_case = self.reporter.add_test_case(
            section="Main Processing",
            subsection="Pipeline",
            title="Process Method",
            description="Test the main processing pipeline",
            preconditions="All component methods work",
            test_data="Sample text",
            steps=[
                "Call process() method",
                "Verify all sub-methods are called"
            ],
            expected_result="Should call read_text_file, create_chunks, and create_faiss_index",
            actual_result="",
            status="RUNNING"
        )
        
        mock_read.return_value = self.sample_text
        mock_create_chunks.return_value = self.sample_docs
        
        try:
            self.test_instance.process()
            
            mock_read.assert_called_once()
            mock_create_chunks.assert_called_once_with(self.sample_text)
            mock_create_index.assert_called_once_with(self.sample_docs)
            
            test_case["ACTUAL RESULT"] = "All methods called correctly"
            test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

    @patch('langchain_community.vectorstores.FAISS.load_local')
    @patch.object(HumanHistoryVectorizer, 'initialize_mistral_model')
    def test_initialize_rag_pipeline(self, mock_init_mistral, mock_load):
        """Test RAG pipeline initialization"""
        test_case = self.reporter.add_test_case(
            section="RAG Pipeline",
            subsection="Initialization",
            title="Initialize RAG Pipeline",
            description="Test RAG pipeline initialization",
            preconditions="FAISS index exists, Mistral API key available",
            test_data="Test index path",
            steps=[
                "Call initialize_rag_pipeline()",
                "Verify components are initialized"
            ],
            expected_result="Should return initialized RetrievalQA instance",
            actual_result="",
            status="RUNNING"
        )
        
        mock_vector_store = MagicMock()
        mock_load.return_value = mock_vector_store
        mock_llm = MagicMock()
        mock_init_mistral.return_value = mock_llm
        
        try:
            result = self.test_instance.initialize_rag_pipeline()
            
            mock_load.assert_called_once_with(
                "test_index", 
                self.test_instance.embeddings, 
                allow_dangerous_deserialization=True
            )
            mock_init_mistral.assert_called_once()
            mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
            self.assertIsNotNone(result)
            
            test_case["ACTUAL RESULT"] = "RetrievalQA instance created"
            test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

    @patch.object(HumanHistoryVectorizer, 'initialize_rag_pipeline')
    def test_query_with_rag(self, mock_init_rag):
        """Test querying with RAG pipeline"""
        test_case = self.reporter.add_test_case(
            section="RAG Pipeline",
            subsection="Querying",
            title="Query with RAG",
            description="Test querying the RAG pipeline",
            preconditions="RAG pipeline initialized",
            test_data="Test question",
            steps=[
                "Call query_with_rag() with test question",
                "Verify response"
            ],
            expected_result="Should return answer from RAG pipeline",
            actual_result="",
            status="RUNNING"
        )
        
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = "test answer"
        mock_init_rag.return_value = mock_pipeline
        
        with patch.object(self.test_instance, '_log_query') as mock_log:
            try:
                answer, _ = self.test_instance.query_with_rag("test question")
                
                self.assertEqual(answer, "test answer")
                mock_init_rag.assert_called_once()
                mock_pipeline.run.assert_called_once_with("test question")
                mock_log.assert_called_once_with("test question", "test answer")
                
                test_case["ACTUAL RESULT"] = "Returned 'test answer'"
                test_case["STATUS"] = "PASS"
            except AssertionError as e:
                test_case["ACTUAL RESULT"] = str(e)
                test_case["STATUS"] = "FAIL"
                raise

    @patch.dict('os.environ', {'MISTRAL_API_KEY': 'test_key'})
    @patch('langchain_mistralai.ChatMistralAI')
    def test_initialize_mistral_model(self, mock_mistral):
        """Test Mistral model initialization"""
        test_case = self.reporter.add_test_case(
            section="Model Initialization",
            subsection="Mistral",
            title="Initialize Mistral Model",
            description="Test Mistral model initialization",
            preconditions="MISTRAL_API_KEY set in environment",
            test_data="API key = 'test_key'",
            steps=[
                "Call initialize_mistral_model()",
                "Verify model initialization"
            ],
            expected_result="Should return initialized ChatMistralAI instance",
            actual_result="",
            status="RUNNING"
        )
        
        try:
            result = self.test_instance.initialize_mistral_model()
            
            mock_mistral.assert_called_once_with(
                model="mistral-large-latest",
                temperature=0.2,
                max_retries=2,
                api_key='test_key'
            )
            self.assertEqual(result, mock_mistral.return_value)
            
            test_case["ACTUAL RESULT"] = "ChatMistralAI instance created"
            test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

    @patch.dict('os.environ', {}, clear=True)
    def test_initialize_mistral_model_no_key(self):
        """Test Mistral model initialization without API key"""
        test_case = self.reporter.add_test_case(
            section="Model Initialization",
            subsection="Mistral",
            title="Initialize Mistral Without API Key",
            description="Test error handling when API key is missing",
            preconditions="MISTRAL_API_KEY not set",
            test_data="No API key in environment",
            steps=[
                "Call initialize_mistral_model() without API key",
                "Catch expected exception"
            ],
            expected_result="Should raise ValueError",
            actual_result="",
            status="RUNNING"
        )
        
        try:
            with self.assertRaises(ValueError):
                self.test_instance.initialize_mistral_model()
            
            test_case["ACTUAL RESULT"] = "Raised ValueError"
            test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

    def test_get_vectorizer(self):
        """Test the helper function"""
        test_case = self.reporter.add_test_case(
            section="Utility Functions",
            subsection="Helper",
            title="Get Vectorizer",
            description="Test the get_vectorizer helper function",
            preconditions="HumanHistoryVectorizer class available",
            test_data="No input",
            steps=[
                "Call get_vectorizer()",
                "Verify returned instance"
            ],
            expected_result="Should return HumanHistoryVectorizer instance",
            actual_result="",
            status="RUNNING"
        )
        
        try:
            result = get_vectorizer()
            self.assertIsInstance(result, HumanHistoryVectorizer)
            
            test_case["ACTUAL RESULT"] = "Returned HumanHistoryVectorizer instance"
            test_case["STATUS"] = "PASS"
        except AssertionError as e:
            test_case["ACTUAL RESULT"] = str(e)
            test_case["STATUS"] = "FAIL"
            raise

if __name__ == '__main__':
    unittest.main()