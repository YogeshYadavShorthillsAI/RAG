import streamlit as st
import faiss
import numpy as np
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from mistralai.client import MistralClient

# Load FAISS index and embeddings
INDEX_PATH = "faiss_index/index.faiss"
EMBEDDINGS_PATH = "faiss_index/embeddings.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MISTRAL_API_KEY = "your_mistral_api_key"

def load_faiss_index():
    index = faiss.read_index(INDEX_PATH)
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return index, data

st.title("Human History QA System")
st.write("Ask any question about human history, and I'll retrieve the most relevant information!")

query = st.text_input("Enter your question:")
if query:
    # Load FAISS index
    index, data = load_faiss_index()
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    query_vector = np.array(embeddings.embed_query(query)).astype('float32').reshape(1, -1)
    
    # Retrieve top results
    k = 3  # Number of results to fetch
    distances, indices = index.search(query_vector, k)
    retrieved_texts = [data[i] for i in indices[0] if i < len(data)]
    
    # Display retrieved context
    st.subheader("Retrieved Context")
    for idx, text in enumerate(retrieved_texts):
        st.write(f"**Chunk {idx+1}:** {text}")
    
    # Query Mistral AI
    client = MistralClient(api_key=MISTRAL_API_KEY)
    response = client.chat(
        model="mistral-large",
        messages=[
            {"role": "system", "content": "You are an expert historian. Answer based on the retrieved context."},
            {"role": "user", "content": "\n".join(retrieved_texts) + f"\nQuestion: {query}"},
        ]
    )
    
    # Display response
    st.subheader("Mistral AI Answer")
    st.write(response["choices"][0]["message"]["content"])
