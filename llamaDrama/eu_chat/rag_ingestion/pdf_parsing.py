# Using llama lite parse
from liteparse import LiteParse
from pathlib import Path
from llama_index.core import Document
import hashlib
from llama_index.core.schema import TransformComponent
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"


class StableNodeID(TransformComponent):
    def __call__(self, nodes, **kwargs):
        net_nodes = len(nodes)
        for index, node in enumerate(nodes):
            source = node.metadata.get("file_name", "unknown")

            raw = source + node.text.strip()

            node.id_ = hashlib.md5(raw.encode()).hexdigest()
            print(f"Processed {index}/{net_nodes}")

        return nodes


def parselite_folder(folder_path: Path, source_type: str, authority_rank: int, jurisdiction: str):
    parser = LiteParse(ocr_enabled=False)

    docs = []

    for file_path in folder_path.glob("*.pdf"):

        result = parser.parse(file_path)

        for page in result.pages:

            text = page.text
            # remove emoty
            if not text or not text.strip():
                continue

            doc = Document(
                text=page.text,
                metadata={
                    "source_type": source_type,
                    "authority_rank": authority_rank,
                    "jurisdiction_label": jurisdiction,
                    "document_id": f"{source_type}::{file_path.name}",
                    "file_name": file_path.name,
                    "page_label": page.page_num
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


def inspect_chunks(documents, splitter):
    nodes = splitter.get_nodes_from_documents(documents)

    lengths = [len(n.text) for n in nodes]

    print("TOTAL CHUNKS:", len(nodes))
    print("MIN LENGTH:", min(lengths))
    print("MAX LENGTH:", max(lengths))
    print("AVG LENGTH:", sum(lengths) / len(lengths))

    print("\nSAMPLE CHUNKS:\n")

    for i, n in enumerate(nodes[:2]):
        print("=" * 80)
        print(f"CHUNK {i}")
        print(n.text[:1200])


if __name__ == "__main__":
    sampler()
