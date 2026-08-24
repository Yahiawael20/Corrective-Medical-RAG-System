import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def clean_text(text: str) -> str:
    """
    Clean extracted text from PDF.
    """

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)

    # Remove page numbers
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # Remove common headers & footers
    patterns = [
        r"(?i)world health organization",
        r"(?i)table of contents",
        r"(?i)contents",
        r"(?i)chapter\s+\d+",
        r"(?i)copyright.*",
        r"©.*",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text)

    # Remove multiple blank lines
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove extra spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def load_documents(data_path="data"):
    """
    Load all PDF files and clean them.
    """

    documents = []

    pdf_files = sorted(Path(data_path).glob("*.pdf"))

    for pdf in pdf_files:

        print(f"Loading {pdf.name}...")

        loader = PyPDFLoader(str(pdf))

        pages = loader.load()

        for page in pages:

            cleaned_text = clean_text(page.page_content)

            # Skip almost empty pages
            if len(cleaned_text) < 80:
                continue

            page.page_content = cleaned_text

            # ===== Clean Metadata =====

            # Keep only file name
            page.metadata["source"] = Path(page.metadata["source"]).name

            # Human-readable page number
            page.metadata["page_number"] = page.metadata["page"] + 1

            documents.append(page)

    print(f"\nLoaded {len(documents)} clean pages.")

    return documents