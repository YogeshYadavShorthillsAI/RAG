import os
import requests
from bs4 import BeautifulSoup


# Function to Combine All Text Files and Save into One File
def combine_texts(input_folder="output_texts", output_file="combined_text.txt"):
    if not os.path.exists(input_folder):
        print("No extracted text files found!")
        return

    combined_text = ""

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                combined_text += file.read() + "\n\n"

    # Save the combined text into a single file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(combined_text)

    print(f"Combined text saved to {output_file}")


def extract_links(url):
    links = []
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    
    if response.status_code != 200:
        print("Failed to retrieve the page")
        return links

    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find search result headings
    results = soup.find_all('div', class_="mw-search-result-heading")
    
    for result in results:
        a_tag = result.find('a', href=True)
        if a_tag:
            title = a_tag.text.strip()
            full_url = "https://en.wikipedia.org" + a_tag['href']
            links.append((title, full_url))

    return links


def extract_text(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    
    if response.status_code != 200:
        print(f"Failed to retrieve {url}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract paragraphs from the main content area
    content_div = soup.find("div", class_="mw-parser-output")  # Corrected class name
    if not content_div:
        return "No content found!"

    paragraphs = content_div.find_all("p")
    
    # Join all paragraph texts
    extracted_text = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

    return extracted_text


def save_text_to_file(title, text):
    # Ensure output directory exists
    os.makedirs("output_texts", exist_ok=True)

    # Sanitize title to create a valid filename
    filename = "".join(c if c.isalnum() or c in " _-" else "_" for c in title) + ".txt"
    
    file_path = os.path.join("output_texts", filename)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)
    
    print(f"Saved: {file_path}")


# Wikipedia search URL
search_url = "https://en.wikipedia.org/w/index.php?limit=250&offset=0&profile=default&search=Prehistory+(Human+Origins)&title=Special:Search&ns0=1"

links = extract_links(search_url)

if links:
    for title, link in links:
        print(f"Processing: {title}\nURL: {link}\n")
        page_text = extract_text(link)

        if page_text.strip():  # Save only if text is extracted
            save_text_to_file(title, page_text)

        print("\n" + "=" * 80 + "\n")
else:
    print("No links found!")
combine_texts()