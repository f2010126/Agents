# Ingestion Pipeline
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from qdrant_client import QdrantClient
from pathlib import Path

from pdf_parsing import parselite_folder, StableNodeID
from constants import MODEL, GOOGLE_API_KEY, COLLECTION

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"

Settings.llm = Gemini(
    model_name=MODEL,
    api_key=GOOGLE_API_KEY
)

Settings.embed_model = GeminiEmbedding(
    model_name="gemini-embedding-001",
    api_key=GOOGLE_API_KEY
)

client = QdrantClient(host="localhost", port=6333)


def load_documents():
    def clean(d):
        text = d.text.replace("\r", " ")
        text = " ".join(text.split())
        return Document(text=text, metadata=d.metadata)

    docs = []
    docs += parselite_folder(DOCS_DIR / "EU_AI_ACT", "eu_ai_act", 1, "EU")
    docs += parselite_folder(DOCS_DIR / "GDPR", "gdpr", 2, "EU")
    docs += parselite_folder(DOCS_DIR /
                             "OTHER_REGULATIONS", "other", 3, "GLOBAL")

    return [clean(d) for d in docs]


def build_index():
    documents = load_documents()

    # splitter = SentenceSplitter(
    #     chunk_size=1024,
    #     chunk_overlap=150
    # )
    # storage_context = StorageContext.from_defaults(
    #     vector_store=vector_store
    # )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
    )

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=1024,
                chunk_overlap=150,
            ),
            StableNodeID(),
            Settings.embed_model,
        ],
        vector_store=vector_store,
    )

    # documents -> chunks -> stable ids -> embeddings -> qdrant upsert
    nodes = pipeline.run(documents=documents, show_progress=True)
    # optional, useful for querying later
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )

    print(f"Ingestion complete. {len(nodes)} nodes processed.")
    print("Ingestion complete → Qdrant populated.")
    return index


if __name__ == "__main__":
    build_index()
