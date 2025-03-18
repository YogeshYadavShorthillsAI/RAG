import os
import re
import numpy as np
import faiss
import google.generativeai as genai
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini API (if using)
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY is missing. Gemini API will not be used.")

# Choose between LLaMA 3 or Gemini (Set to True to use Gemini API)
USE_GEMINI = True if api_key else False  

# Function to Read Combined Text File
def read_combined_text(file_path="combined_text.txt"):
    if not os.path.exists(file_path):
        print("⚠️ Combined text file not found! Run scrape.py first.")
        return ""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Step 1: Preprocess Text
def preprocess(text):
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"\[\d+\]", "", text)  # Remove citations
    return text.strip()

# Step 2: Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

# Step 3: Embed Chunks
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(chunks):
    return np.array(embedding_model.encode(chunks)).astype("float32")

# Step 4: Store in FAISS
def create_faiss_index(embeddings):
    if embeddings.size == 0:
        print("⚠️ No embeddings found!")
        return None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Step 5: Retrieve Relevant Chunks
def retrieve_chunks(query, index, chunks, k=3):
    if index is None:
        return []
    query_embedding = np.array(embedding_model.encode([query])).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# **LLaMA 3 Setup (if NOT using Gemini)**
if not USE_GEMINI:
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# **Function to Generate Answers using Gemini API**
def generate_answer_gemini(query, retrieved_chunks):
    if not retrieved_chunks:
        return "⚠️ No relevant context found. Try rephrasing your query."

    context = " ".join(retrieved_chunks)
    prompt = f"Context: {context} \n\nQuestion: {query} \n\nAnswer:"

    try:
        response = genai.GenerativeModel("gemini-1.5-flash-latest").generate_content(prompt)
        return response.text if hasattr(response, "text") else "⚠️ No valid response received."
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

# **Function to Generate Answers using LLaMA 3**
def generate_answer_llama(query, retrieved_chunks):
    if not retrieved_chunks:
        return "⚠️ No relevant context found. Try rephrasing your query."

    context = " ".join(retrieved_chunks)
    prompt = f"Context: {context} \n\nQuestion: {query} \n\nAnswer:"

    response = generator(prompt, max_length=512, temperature=0.7, do_sample=True)
    return response[0]['generated_text']

# **Main Execution**
if __name__ == "__main__":
    combined_text = read_combined_text()
    
    if not combined_text.strip():
        print("❌ No content available for processing. Exiting...")
        exit()

    processed_text = preprocess(combined_text)
    chunks = text_splitter.split_text(processed_text)
    embeddings = get_embeddings(chunks)
    index = create_faiss_index(embeddings)

    while True:
        query = input("\n🔍 Enter your query (or type 'exit' to stop): ").strip()
        if query.lower() == "exit":
            print("👋 Exiting program. Goodbye!")
            break

        retrieved_chunks = retrieve_chunks(query, index, chunks)

        # Choose the model for response generation
        if USE_GEMINI:
            answer = generate_answer_gemini(query, retrieved_chunks)
        else:
            answer = generate_answer_llama(query, retrieved_chunks)

        print("\n📝 Generated Answer:\n", answer)