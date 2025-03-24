import os
import requests
from bs4 import BeautifulSoup


class WikipediaScraper:
    """Handles extracting links and text from Wikipedia search results."""
    
    BASE_URL = "https://en.wikipedia.org"

    def __init__(self, search_url):
        self.search_url = search_url

    def extract_links(self):
        """Extracts Wikipedia article links from a search page."""
        links = []
        response = requests.get(self.search_url, headers={"User-Agent": "Mozilla/5.0"})
        
        if response.status_code != 200:
            print("Failed to retrieve the search results page.")
            return links

        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.find_all('div', class_="mw-search-result-heading")

        for result in results:
            a_tag = result.find('a', href=True)
            if a_tag:
                title = a_tag.text.strip()
                full_url = self.BASE_URL + a_tag['href']
                links.append((title, full_url))

        return links

    def extract_text(self, url):
        """Extracts the main text content from a Wikipedia page."""
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        
        if response.status_code != 200:
            print(f"Failed to retrieve {url}")
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", class_="mw-parser-output")
        if not content_div:
            return "No content found!"

        paragraphs = content_div.find_all("p")
        extracted_text = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

        return extracted_text


class FileHandler:
    """Handles saving and combining text files."""

    def __init__(self, output_folder="output_texts", combined_file="combined_text.txt"):
        self.output_folder = output_folder
        self.combined_file = combined_file
        os.makedirs(self.output_folder, exist_ok=True)

    def save_text_to_file(self, title, text):
        """Saves extracted text to a file."""
        filename = "".join(c if c.isalnum() or c in " _-" else "_" for c in title) + ".txt"
        file_path = os.path.join(self.output_folder, filename)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)

        print(f"Saved: {file_path}")

    def combine_texts(self):
        """Combines all extracted text files into a single file."""
        if not os.path.exists(self.output_folder):
            print("No extracted text files found!")
            return

        combined_text = ""

        for filename in os.listdir(self.output_folder):
            file_path = os.path.join(self.output_folder, filename)

            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    combined_text += file.read() + "\n\n"

        with open(self.combined_file, "w", encoding="utf-8") as file:
            file.write(combined_text)

        print(f"Combined text saved to {self.combined_file}")


class ScraperRunner:
    """Orchestrates Wikipedia scraping and file handling."""

    def __init__(self, search_url):
        self.scraper = WikipediaScraper(search_url)
        self.file_handler = FileHandler()

    def run(self):
        """Runs the scraping process and saves extracted text."""
        links = self.scraper.extract_links()

        if not links:
            print("No links found!")
            return

        for title, link in links:
            print(f"Processing: {title}\nURL: {link}\n")
            page_text = self.scraper.extract_text(link)

            if page_text.strip():
                self.file_handler.save_text_to_file(title, page_text)

            print("\n" + "=" * 80 + "\n")

        self.file_handler.combine_texts()


# Wikipedia search URL
search_url = "https://en.wikipedia.org/w/index.php?limit=250&offset=0&profile=default&search=Prehistory+(Human+Origins)&title=Special:Search&ns0=1"

# Run the scraper
scraper = ScraperRunner(search_url)
scraper.run()
