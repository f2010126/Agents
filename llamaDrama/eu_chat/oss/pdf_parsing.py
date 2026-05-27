# Using llama lite parse
from liteparse import LiteParse
from pathlib import Path
from llama_index.core import Document

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"


def parselite_folder(folder_path: Path, source_type: str, authority_rank: int, jurisdiction: str):
    parser = LiteParse()

    docs = []

    for file_path in folder_path.glob("*.pdf"):

        result = parser.parse(file_path)

        for page in result.pages:

            doc = Document(
                text=page.text,
                metadata={
                    "source_type": source_type,
                    "authority_rank": authority_rank,
                    "jurisdiction_label": jurisdiction,
                    "document_id": f"{source_type}::{file_path.name}",
                    "file_name": file_path.name,
                    "page_label": page.page_number
                }
            )

            docs.append(doc)

    return docs


def sampler():
    parser = LiteParse()
    result = parser.parse(DOCS_DIR/"GDPR/GDPR.pdf")

    # Full document text
    # print(result.text)
    print(result.pages[0].text[:1500])

    # Per-page data
    # for page in result.pages:
    #     print(f"Page {page.page_num}: {len(page.text_items)} text items")


if __name__ == "__main__":
    sampler()
