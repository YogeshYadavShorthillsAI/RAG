import os
import json
import time
import requests
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from tqdm import tqdm
from dotenv import load_dotenv
from bert_score import score as bert_score

load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
LOG_FILE = "qa_interactions.log"
RESULTS_FILE = "evaluation_results.json"
SUMMARY_FILE = "evaluation_summary.json"
TEST_CASES_FILE = "test_cases.json"
VALID_MODELS = ["mistral-7b", "mistral-tiny", "gpt-4"]
MISTRAL_MODEL = "mistral-tiny"

METRIC_WEIGHTS = {
    "rouge_score": 0.25,
    "cosine_similarity": 0.25,
    "bert_score_f1": 0.5
}

if MISTRAL_MODEL not in VALID_MODELS:
    raise ValueError(f"Invalid model '{MISTRAL_MODEL}'. Choose from {VALID_MODELS}")

# Initialize models
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def setup_logger():
    if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
        with open(LOG_FILE, "w") as f:
            json.dump([], f)  # Ensures a valid JSON array


def log_interaction(entry):
    try:
        if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
            logs = []
        else:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)  # Try loading existing data

        logs.append(entry)  # Add new entry

        with open(LOG_FILE, "w") as f:
            json.dump(logs, f, indent=4)  # Save updated logs
    except json.JSONDecodeError:  # Handle corrupted JSON
        with open(LOG_FILE, "w") as f:
            json.dump([entry], f, indent=4)  # Reset with the first entry


def calculate_metrics(generated, reference):
    emb_gen = similarity_model.encode(generated)
    emb_ref = similarity_model.encode(reference)
    cosine_sim = np.dot(emb_gen, emb_ref) / (np.linalg.norm(emb_gen) * np.linalg.norm(emb_ref))
    rouge_score = rouge.score(reference, generated)['rougeL'].fmeasure
    bert_precision, bert_recall, bert_f1 = bert_score([generated], [reference], lang="en", model_type="bert-base-uncased")
    bert_f1 = bert_f1.mean().item()
    final_score = (
        METRIC_WEIGHTS["rouge_score"] * rouge_score +
        METRIC_WEIGHTS["cosine_similarity"] * float(cosine_sim) +
        METRIC_WEIGHTS["bert_score_f1"] * bert_f1
    )
    return {
        "rouge_score": rouge_score,
        "cosine_similarity": float(cosine_sim),
        "bert_score_f1": bert_f1,
        "final_score": final_score
    }

def load_test_cases(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def qa_pipeline(question, context):
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
    for attempt in range(5):
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        time.sleep(2 ** attempt)
    return "Error generating response"

def process_test_cases():
    setup_logger()
    test_cases = load_test_cases(TEST_CASES_FILE)
    if not test_cases:
        print("No test cases found.")
        return
    results = []
    pbar = tqdm(total=len(test_cases), desc="Processing test cases")
    for case in test_cases:
        generated = qa_pipeline(case["question"], case["context"])  # Use lowercase keys
        metrics = calculate_metrics(generated, case["answer"])
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": case["question"],
            "context": case["context"],
            "generated_answer": generated,
            "reference_answer": case["answer"],
            **metrics
        }
        log_interaction(log_entry)
        results.append(log_entry)
        pbar.update(1)
    pbar.close()
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {RESULTS_FILE}")
    return results

def calculate_grade(score):
    if score >= 0.90: return "A (Excellent)"
    elif score >= 0.80: return "B (Good)"
    elif score >= 0.70: return "C (Average)"
    elif score >= 0.60: return "D (Below Average)"
    else: return "F (Poor)"

if __name__ == "__main__":
    results = process_test_cases()
    if results:
        summary = {key: np.mean([r[key] for r in results]) for key in ["rouge_score", "cosine_similarity", "bert_score_f1", "final_score"]}
        summary["grade"] = calculate_grade(summary["final_score"])
        with open(SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Summary metrics saved to {SUMMARY_FILE}")