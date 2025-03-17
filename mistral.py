import os
import json
import textwrap
import time
from tqdm import tqdm
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# 🔹 Load API Key Securely
MISTRAL_API_KEY = "ZxBa8fO4NjkbrWNID4QgUYtyJy84nh0K"
# 🔹 Initialize Mistral Client
client = MistralClient(api_key=MISTRAL_API_KEY)

# 🔹 Function to load scraped data
def load_scraped_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# 🔹 Function to split text into chunks
def split_into_chunks(text, chunk_size=500):
    return textwrap.wrap(text, chunk_size)

# 🔹 Function to generate test cases (question + answer)
def generate_test_case(context, retries=3):
    prompt = f"""
    Based on the following machine learning content, generate a test question and provide a detailed answer:

    {context}

    Format the response strictly as:
    Q: [Generated question]
    A: [Generated answer]
    """

    for attempt in range(retries):
        try:
            response = client.chat(
                model="mistral-tiny",
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.7,
                max_tokens=150
            )

            generated_text = response.choices[0].message.content.strip()

            # Extract Q & A
            question, answer = "", ""
            lines = generated_text.split("\n")
            for line in lines:
                if line.startswith("Q:"):
                    question = line[2:].strip()
                elif line.startswith("A:"):
                    answer = line[2:].strip()

            if question and answer:
                return question, answer
            else:
                raise ValueError("Invalid response format")

        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(2)  # Wait before retrying

    return None, None  # Return empty values on failure

# 🔹 Load & process data
scraped_text = load_scraped_data("combined_text.txt")
chunks = split_into_chunks(scraped_text, 500)

# 🔹 Load existing progress
output_file = "generated_test_cases.json"
test_cases = []

try:
    with open(output_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    pass  # Start fresh if no valid file exists

start_index = len(test_cases)
print(f"▶️ Resuming from index {start_index}")

# 🔹 Generate test cases
for i in tqdm(range(start_index, min(start_index + 1000, len(chunks)))):
    question, answer = generate_test_case(chunks[i])
    if question and answer:
        test_cases.append({"context": chunks[i], "question": question, "answer": answer})

    # ✅ Save progress every 10 test cases
    if i % 10 == 0:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(test_cases, f, indent=4)

# 🔹 Final Save
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(test_cases, f, indent=4)

print("✅ Test cases (questions + answers) generated successfully!")
