from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os
import re
import google.generativeai as genai

import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Function to Read Combined Text File
def read_combined_text(file_path="combined_text.txt"):
    if not os.path.exists(file_path):
        print("Combined text file not found! Run scrape.py first.")
        return ""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Step 1: Preprocess Text
def preprocess(text):
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"\[\d+\]", "", text)  # Remove citations
    return text

# Step 2: Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

# Step 3: Embed Chunks
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(chunks):
    return np.array(embedding_model.encode(chunks)).astype("float32")

# Step 4: Store in FAISS
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Step 5: Retrieve Relevant Chunks
def retrieve_chunks(query, index, chunks):
    query_embedding = np.array(embedding_model.encode([query])).astype("float32")
    distances, indices = index.search(query_embedding, k=3)
    return [chunks[i] for i in indices[0]]

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Define the model to use
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  # Change to "meta-llama/Llama-2-7b-chat-hf" if needed

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_answer(query, retrieved_chunks):
    context = " ".join(retrieved_chunks)
    prompt = f"Context: {context} \n\nQuestion: {query} \n\nAnswer:"

    response = generator(prompt, max_length=512, temperature=0.7, do_sample=True)
    return response[0]['generated_text']


# Running the RAG Pipeline
if __name__ == "__main__":
    combined_text = read_combined_text()  # Load combined text file
    query = ''
    while query!='exit':
        if combined_text.strip():
            processed_text = preprocess(combined_text)
            chunks = text_splitter.split_text(processed_text)
            embeddings = get_embeddings(chunks)
            index = create_faiss_index(embeddings)

            query = str(input("Enter your query here: "))
            retrieved_chunks = retrieve_chunks(query, index, chunks)
            
            # Fixed variable issue
            answer = generate_answer(query, retrieved_chunks)
            print("\nGenerated Answer:", answer)
        else:
            print("No content available for processing.")
