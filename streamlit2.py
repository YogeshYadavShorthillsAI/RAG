import os
import re
import json
import numpy as np
import faiss
import torch
import textwrap
import time
from datetime import datetime
from tqdm import tqdm
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = MistralClient(api_key=MISTRAL_API_KEY)

# Function to Read Combined Text File
def read_combined_text(file_path="combined_text.txt"):
    if not os.path.exists(file_path):
        print("⚠️ Combined text file not found! Run scrape.py first.")
        return ""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Preprocessing
def preprocess(text):
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"\[\d+\]", "", text)  # Remove citations
    return text.strip()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(chunks):
    return np.array(embedding_model.encode(chunks)).astype("float32")

def create_faiss_index(embeddings):
    if embeddings.size == 0:
        print("⚠️ No embeddings found!")
        return None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_chunks(query, index, chunks, k=3):
    if index is None:
        return []
    query_embedding = np.array(embedding_model.encode([query])).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# Function to log queries and responses in a JSON file
def log_query_answer(question, answer, log_file="query_log.json"):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer
    }
    
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as file:
            try:
                logs = json.load(file)
                if not isinstance(logs, list):
                    logs = []
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
    
    logs.append(log_entry)

    with open(log_file, "w", encoding="utf-8") as file:
        json.dump(logs, file, indent=4, ensure_ascii=False)

# Function to generate answer using Mistral AI and log it
def generate_answer_mistral(query, retrieved_chunks):
    if not retrieved_chunks:
        answer = "⚠️ No relevant context found. Try rephrasing your query."
        log_query_answer(query, answer)
        return answer

    context = " ".join(retrieved_chunks)
    prompt = f"""
    Context: {context}
    
    Question: {query}
    
    Answer:
    """
    try:
        response = client.chat(
            model="mistral-tiny",
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=0.7,
            max_tokens=150
        )
        answer = response.choices[0].message.content.strip()
        log_query_answer(query, answer)  # Log the query and response
        return answer
    except Exception as e:
        error_msg = f"⚠️ Error generating response: {str(e)}"
        log_query_answer(query, error_msg)
        return error_msg

# Streamlit UI
def main():
    st.title("📚 RAG-Based QA System with Mistral AI")
    option = st.sidebar.radio("Select an Option", ["Start", "Process Data", "Ask Query"])
    
    if option == "Start":
        st.header("🚀 Welcome!")
        st.markdown("""
        1. **Process Data**: Convert text into embeddings & store in FAISS.
        2. **Ask Query**: Retrieve & answer questions using AI.
        """)
    
    elif option == "Process Data":
        st.header("⚙️ Process Data")
        if st.button("Start Processing"):
            combined_text = read_combined_text()
            if not combined_text.strip():
                st.error("⚠️ No content available. Please scrape data first.")
            else:
                st.write("🔹 **Preprocessing Text**...")
                processed_text = preprocess(combined_text)
                chunks = text_splitter.split_text(processed_text)
                st.write("🔹 **Generating Embeddings**...")
                embeddings = get_embeddings(chunks)
                st.write("🔹 **Creating FAISS Index**...")
                index = create_faiss_index(embeddings)
                faiss.write_index(index, "faiss_index.bin")
                st.session_state["faiss_index"] = index
                st.success("✅ Data processed & indexed successfully!")
    
    elif option == "Ask Query":
        st.header("💬 Ask a Question")
        query = st.text_input("Enter your query:")
        if st.button("Get Answer"):
            if not os.path.exists("faiss_index.bin"):
                st.error("⚠️ No processed data! Process data first.")
            else:
                index = faiss.read_index("faiss_index.bin")
                with open("combined_text.txt", "r", encoding="utf-8") as file:
                    processed_text = file.read()
                chunks = text_splitter.split_text(processed_text)
                retrieved_chunks = retrieve_chunks(query, index, chunks)
                answer = generate_answer_mistral(query, retrieved_chunks)
                st.success("✅ Answer Generated!")
                st.write(answer)

if __name__ == "__main__":
    main()
