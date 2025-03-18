Here is your full `README.md` file. You can copy and paste it directly into your project.

```markdown
# Retrieval-Augmented Generation (RAG) Based QA System  

## Overview  
This project implements a **Retrieval-Augmented Generation (RAG)** based **Question Answering (QA)** system using the **Mistral AI model**. It leverages **FAISS (Facebook AI Similarity Search)** for efficient vector retrieval and **Sentence Transformers** for text embeddings. The system allows users to query a knowledge base and receive AI-generated answers based on relevant retrieved content.  

---

## Features  
- **Text Preprocessing**: Cleans and processes scraped text.  
- **Chunking**: Splits large text into manageable segments.  
- **Vector Embeddings**: Converts text into high-dimensional embeddings using Sentence Transformers.  
- **FAISS Indexing**: Enables efficient similarity search for retrieval.  
- **RAG-based Answering**: Uses retrieved context with the Mistral AI model for generating responses.  
- **Multiple LLM Support**: Includes Mistral-based and Gemini-based RAG implementations.  
- **Streamlit UI**: Provides an interactive interface for querying data and logging user interactions.  
- **Evaluation Metrics**: Measures accuracy using Exact Match, Cosine Similarity, ROUGE-L, and BERT-based scores.  

---

## Folder Structure  
```
📂 RAG/  
│── 📂 output_texts/        # Stores processed text outputs  
│── 📜 .env                 # Environment variables (API keys, configurations)  
│── 📜 chunked_text.json    # Contains chunked versions of combined text  
│── 📜 combined_text.txt    # Raw scraped text for processing  
│── 📜 evaluation_results.csv  # RAG evaluation results  
│── 📜 evaluation_results_1000.csv  # Extended evaluation results  
│── 📜 evaluation.py        # Implements evaluation metrics for RAG performance  
│── 📜 faiss_index.bin      # FAISS binary index file  
│── 📜 faiss_index.index    # FAISS index for vector search  
│── 📜 gemini_rag.py        # Gemini AI-based RAG implementation  
│── 📜 generated_test_cases.json  # Test cases generated for evaluation  
│── 📜 generated_test_cases.txt   # Text-based test cases  
│── 📜 main3.py             # Main script for executing the pipeline  
│── 📜 mistral_test_generation.py  # Mistral AI test case generation  
│── 📜 qa_interactions.log  # Logs user queries and responses  
│── 📜 queries.csv          # Logged queries for analysis  
│── 📜 query_log.json       # JSON log for query interactions  
│── 📜 rag_evaluation_results.csv  # Evaluation scores for the system  
│── 📜 requirements.txt     # List of dependencies for the project  
│── 📜 scrape.py            # Web scraping script for text extraction  
│── 📜 streamlit2.py        # Streamlit-based UI for RAG system  
│── 📜 test_queries.json    # Test queries for validation  
│── 📂 venv/                # Virtual environment for dependencies  
```

---

## Installation  
### 1️⃣ Clone the repository  
```bash
git clone <repository_url>
cd RAG
```

### 2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Set up environment variables  
Create a `.env` file and add:  
```
MISTRAL_API_KEY=your_api_key_here
GEMINI_API_KEY=your_api_key_here
```

---

## Usage  
### 🔹 Run Preprocessing & Indexing  
```bash
python main3.py
```

### 🔹 Run Streamlit UI  
```bash
streamlit run streamlit2.py
```

---

## RAG Evaluation  
The evaluation system measures:  
- **Exact Match**: Checks if generated and reference answers are identical.  
- **Cosine Similarity**: Measures vector-based similarity of answers.  
- **ROUGE-L Score**: Evaluates overlap between reference and generated answers.  
- **BERT-Based Similarity**: Uses BERT embeddings for semantic similarity.  

Run evaluation:  
```bash
python evaluation.py
```

---

## Testing  
- **Testing RAG Pipeline**: Validate text retrieval, embedding generation, and response accuracy using Mistral and Gemini AI.  
- **Testing Evaluation Metrics**: Ensure reliability of Exact Match, Cosine Similarity, ROUGE-L, and BERT-based scores.  

---

## Documentation  
For detailed documentation, follow the link:  
[Click here to access the documentation](https://shorthillstech.sharepoint.com/:fl:/g/contentstorage/CSP_140196ef-9842-4f78-bfac-4aea9a3944e6/Ea5vcR4_mL1Eg3qwmeN50PUB-2_f1m3Yql4WFJ6YvjfSFw?e=RcbD1V&nav=cz0lMkZjb250ZW50c3RvcmFnZSUyRkNTUF8xNDAxOTZlZi05ODQyLTRmNzgtYmZhYy00YWVhOWEzOTQ0ZTYmZD1iJTIxNzVZQkZFS1llRS1fckVycW1qbEU1bmtyU0hGdEh6VlByZTU3UUVsdE1XemVad3JBRlVEVlNabWdRVFBmb2tCayZmPTAxSVhDNEhEVk9ONVlSNFA0WVhWQ0lHNlZRVEhSWFRVSFYmYz0lMkYmYT1Mb29wQXBwJnA9JTQwZmx1aWR4JTJGbG9vcC1wYWdlLWNvbnRhaW5lciZ4PSU3QiUyMnclMjIlM0ElMjJUMFJUVUh4emFHOXlkR2hwYkd4emRHVmphQzV6YUdGeVpYQnZhVzUwTG1OdmJYeGlJVGMxV1VKR1JVdFpaVVV0WDNKRmNuRnRhbXhGTlc1cmNsTklSblJJZWxaUWNtVTFOMUZGYkhSTlYzcGxXbmR5UVVaVlJGWlRXbTFuVVZSUVptOXJRbXQ4TURGSldFTTBTRVJZUmtkRVVrWlVXVFJUUkZKSVdsVlZSbEZJTTFnMFFqWktWUSUzRCUzRCUyMiUyQyUyMmklMjIlM0ElMjJiNGJhZDZiOC00MDgxLTRkMjQtYjY1YS02YjdjZjU1ODE4ZTklMjIlN0Q%3D)  

---

## Contributions  
Feel free to open issues and submit PRs for improvements!  

---

## License  
This project is licensed under [MIT License](LICENSE).  
```

This `README.md` file is fully formatted and ready to be pasted into your project. Let me know if you need any modifications! 🚀
