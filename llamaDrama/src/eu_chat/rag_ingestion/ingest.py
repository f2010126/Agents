# Ingestion Pipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TransformComponent
from qdrant_client import QdrantClient
from pathlib import Path

from llamaDrama.src.eu_chat.rag_ingestion.pdf_parsing import parselite_folder, StableNodeID
# my constants. WIll vary with provider
from llamaDrama.src.eu_chat.constants import COLLECTION
# init what models llamaindex needs to use
from llamaDrama.src.eu_chat.init_llms import init_models

init_models()

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"

client = QdrantClient(host="localhost", port=6333)

# Qdrant Schema


def ensure_qdrant_collection():
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION not in collections:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=3072,
                distance=Distance.COSINE
            )
        )

    # Enforce Payload index
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="jurisdiction_label",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="source_type",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="authority_rank",
        field_schema=PayloadSchemaType.INTEGER,
    )

    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="document_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="page_label",
        field_schema=PayloadSchemaType.INTEGER,
    )

# Set Payload Schema


class MetadataNormalizer(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for n in nodes:
            if n.metadata:
                n.metadata = build_payload(n.metadata)
        return nodes


def build_payload(metadata: dict):
    return {
        "source_type": metadata.get("source_type"),
        "jurisdiction_label": metadata.get("jurisdiction_label"),
        "authority_rank": metadata.get("authority_rank"),
        "document_id": metadata.get("document_id"),
        "page_label": metadata.get("page_label"),
        "file_name": metadata.get("file_name"),  # needed for the tool.
    }


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
            MetadataNormalizer(),
            StableNodeID(),
            Settings.embed_model,
        ],
        vector_store=vector_store,
    )

    # documents -> chunks -> stable ids -> embeddings -> qdrant upsert
    nodes = pipeline.run(documents=documents, show_progress=True)

    # build query index wrapper
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )
    print(
        f"Ingestion complete. {len(nodes)} nodes processed. \n Ingestion complete → Qdrant populated.")
    return index


if __name__ == "__main__":
    ensure_qdrant_collection()
    build_index()
