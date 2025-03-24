from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain.schema import Document
import faiss
import os
import json
import uuid
from datetime import datetime
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()

class HumanHistoryVectorizer:
    """Process human history data into a FAISS vector database and implement RAG pipeline."""
    
    def __init__(
        self, 
        text_file="combined_text.txt", 
        chunk_size=1000, 
        chunk_overlap=200,
        faiss_index_path="faiss_index"
    ):
        """
        Initialize the vectorizer.
        
        Args:
            text_file: Path to the pre-combined text file.
            chunk_size: Size of text chunks in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.
            faiss_index_path: Directory to store FAISS index.
        """
        self.text_file = text_file
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.faiss_index_path = faiss_index_path
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True
        )
    
    def read_text_file(self) -> str:
        """Read text from the specified file."""
        if not os.path.exists(self.text_file):
            raise FileNotFoundError(f"File {self.text_file} not found.")
        
        with open(self.text_file, 'r', encoding='utf-8') as file:
            return file.read()
    
    def create_chunks(self, text: str) -> List[Document]:
        """Split text into overlapping chunks."""
        print(f"Creating chunks with size {self.chunk_size} and overlap {self.chunk_overlap}...")
        documents = self.text_splitter.create_documents([text])
        for doc in documents:
            doc.metadata["chunk_id"] = str(uuid.uuid4())
        print(f"Created {len(documents)} chunks.")
        return documents
    
    def create_faiss_index(self, documents: List[Document]) -> None:
        """Create FAISS index from documents."""
        print("Creating FAISS vector store...")
        vector_store = FAISS.from_documents(documents, self.embeddings)
        vector_store.save_local(self.faiss_index_path)
        print(f"FAISS index stored at {self.faiss_index_path}")
    
    def process(self) -> None:
        """Main processing function to execute the pipeline."""
        text = self.read_text_file()
        documents = self.create_chunks(text)
        self.create_faiss_index(documents)
        print("Processing completed!")
    
    def initialize_rag_pipeline(self):
        """Initialize the RAG pipeline."""
        vector_store = FAISS.load_local(self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)

        mistral_llm = self.initialize_mistral_model()
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a historian AI. Answer the user's question using ONLY the provided context. "
                "If unsure, say so. Provide clear and factual explanations.\n\n"
                "Also greet the user and ask if they have any other questions.\n\n"
                "Context: {context}\n\n"
                "Question: {question}"
            )
        )
        
        return RetrievalQA.from_chain_type(
            llm=mistral_llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt_template}
        )
    
    def query_with_rag(self, query_text: str) -> Tuple[str, str]:
        """Query FAISS and generate an answer using RAG."""
        rag_pipeline = self.initialize_rag_pipeline()
        result = rag_pipeline.run(query_text)
        self._log_query(query_text, result)
        return result, ""
    
    def initialize_mistral_model(self):
        """Initialize Mistral model."""
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY in environment variables.")
        
        return ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.2,
            max_retries=2,
            api_key=mistral_api_key
        )
    
    def _log_query(self, query: str, answer: str) -> None:
        """Log queries and answers."""
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
    def main_menu(self):
        """Displays a menu for the user to choose actions."""
        while True:
            print("\n==== RAG Pipeline Menu ====")
            print("1. Vectorize & Embed (Reprocess Data)")
            print("2. Load Existing FAISS Index & Perform Q/A")
            print("3. Delete Existing FAISS Index & Recreate")
            print("4. Exit")

            choice = input("Enter your choice (1-4): ").strip()

            if choice == "1":
                self.process()
            elif choice == "2":
                if not os.path.exists(self.faiss_index_path):
                    print("FAISS index not found! Please run option 1 first.")
                else:
                    self.run_qna()  # Function to handle user questions
            elif choice == "3":
                if os.path.exists(self.faiss_index_path):
                    os.remove(self.faiss_index_path)
                    print("FAISS index deleted.")
                self.process()
            elif choice == "4":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 4.")

    def run_qna(self):
        """Handles Q/A after loading FAISS."""
        vector_store = FAISS.load_local(
            self.faiss_index_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        while True:
            question = input("\nEnter your question (or type 'exit' to go back): ").strip()
            if question.lower() == "exit":
                break
            answer, _ = self.query_with_rag(question)
            print("\nAnswer:", answer)

# Example usage:
vectorizer = HumanHistoryVectorizer()
vectorizer.main_menu()

