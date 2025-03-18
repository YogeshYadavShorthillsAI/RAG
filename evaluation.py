import os
import pandas as pd
import re
import json
import time
import requests
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from evaluate import load  # For exact match
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel
import torch
from bert_score import score as bert_score  
from sklearn.metrics import precision_score  

load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
LOG_FILE = "qa_interactions_new.log"
RESULTS_FILE = "evaluation_resultnew.csv"
TEST_CASES_FILE = "generated_test_cases.json"  
VALID_MODELS = ["mistral-7b", "mistral-tiny", "gpt-4"]
MISTRAL_MODEL = "mistral-tiny"

if MISTRAL_MODEL not in VALID_MODELS:
    raise ValueError(f"Invalid model '{MISTRAL_MODEL}'. Choose from {VALID_MODELS}")

# Initialize models
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
exact_match_metric = load("exact_match")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def setup_logger():
    """Initialize logging file"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,context,question,generated_answer,reference_answer,exact_match,cosine_similarity,rouge_score,bert_similarity,bert_score_precision,bert_score_recall,bert_score_f1,precision_score,accuracy_score,final_score\n")

def log_interaction(context, question, generated, reference, metrics):
    """Log interaction with timestamp"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "context": context,
        "question": question,
        "generated_answer": generated,
        "reference_answer": reference,
        **metrics
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def calculate_bert_similarity(text1, text2):
    """Calculate BERT-based semantic similarity"""
    inputs = bert_tokenizer([text1, text2], return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
    return cosine_sim

def calculate_metrics(generated, reference):
    """Calculate evaluation metrics"""
    exact_match = exact_match_metric.compute(predictions=[generated], references=[reference])["exact_match"]

    emb_gen = similarity_model.encode(generated)
    emb_ref = similarity_model.encode(reference)
    cosine_sim = np.dot(emb_gen, emb_ref) / (np.linalg.norm(emb_gen) * np.linalg.norm(emb_ref))

    rouge_score = rouge.score(reference, generated)['rougeL'].fmeasure
    bert_sim = calculate_bert_similarity(generated, reference)

    bert_score_precision, bert_score_recall, bert_score_f1 = bert_score(
        [generated], [reference], lang="en", model_type="bert-base-uncased"
    )
    bert_score_precision = bert_score_precision.mean().item()
    bert_score_recall = bert_score_recall.mean().item()
    bert_score_f1 = bert_score_f1.mean().item()

    precision = precision_score(
        [1 if token in generated.split() else 0 for token in reference.split()],
        [1] * len(reference.split()),
        zero_division=0
    )

    accuracy = sum(1 for gt, rt in zip(generated.split(), reference.split()) if gt == rt) / len(reference.split()) if reference.split() else 0.0

    final_score = (
        exact_match * 0.15 +
        cosine_sim * 0.15 +
        rouge_score * 0.15 +
        bert_sim * 0.15 +
        bert_score_f1 * 0.15 +
        precision * 0.15 +
        accuracy * 0.10
    )

    return {
        "exact_match": exact_match,
        "cosine_similarity": float(cosine_sim),
        "rouge_score": rouge_score,
        "bert_similarity": bert_sim,
        "bert_score_precision": bert_score_precision,
        "bert_score_recall": bert_score_recall,
        "bert_score_f1": bert_score_f1,
        "precision_score": precision,
        "accuracy_score": accuracy,
        "final_score": final_score
    }

def load_test_cases(filepath):
    """Load test cases from JSON"""
    try:
        with open(filepath, "r") as file:
            test_cases = json.load(file)
        df = pd.DataFrame(test_cases)
        if not {"context", "question", "answer"}.issubset(df.columns):
            raise ValueError("JSON format is incorrect. Ensure it contains 'context', 'question', and 'answer'.")
        print(f"✅ Loaded {len(df)} test cases.")
        return df
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return pd.DataFrame()

def qa_pipeline(question, context=""):
    """Query Mistral API with retries"""
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": "You are an AI assistant helping with RAG-based QA."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }

    max_retries = 5
    for attempt in range(max_retries):
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        elif response.status_code == 429:  
            wait_time = 2 ** attempt
            print(f"⚠️ Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return "Error generating response"
    print("🚨 Max retries reached. Skipping question.")
    return "Error generating response"

def process_test_cases():
    """Main evaluation process"""
    setup_logger()
    df = load_test_cases(TEST_CASES_FILE)
    
    if df.empty:
        print("⚠️ No test cases found. Check file format.")
        return

    # df = df.head(10)  # Process only first 10 cases for testing

    pbar = tqdm(total=len(df), desc="Processing test cases")
    for idx, row in df.iterrows():
        try:
            generated = qa_pipeline(row["question"], row["context"])
            metrics = calculate_metrics(generated, row["answer"])
            log_interaction(row["context"], row["question"], generated, row["answer"], metrics)

            result_row = pd.DataFrame([{**row, **metrics}])
            result_row.to_csv(RESULTS_FILE, mode="a", header=False, index=False)

            pbar.update(1)
            pbar.set_postfix({"Score": f"{metrics['final_score']:.2f}"})
        except Exception as e:
            print(f"❌ Error processing case {idx}: {e}")
            continue

    pbar.close()
    print(f"\n✅ Processing complete! Results saved to {RESULTS_FILE}.")

if __name__ == "__main__":
    process_test_cases()
