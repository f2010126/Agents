from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.query_engine import RetrieverQueryEngine
import os
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Document,
    load_index_from_storage,
    Settings,
)
from oss.data_cleaning import inspect_raw_documents, clean_text
from oss.pdf_parsing import parselite_folder, inspect_chunks

# add your GOOGLE API key here
MODEL = "models/gemini-2.5-flash"

PERSIST_DIR = "./storage"
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "oss" / "docs"

Settings.llm = Gemini(
    model_name=MODEL,
    api_key=GOOGLE_API_KEY
)

Settings.embed_model = GeminiEmbedding(
    model_name="gemini-embedding-001",  # models/text-embedding-001
    api_key=GOOGLE_API_KEY
)

client = QdrantClient(host="localhost", port=6333)


def load_documents():

    eu_docs = parselite_folder(
        DOCS_DIR / "EU_AI_ACT",
        source_type="eu_ai_act",
        authority_rank=1,
        jurisdiction="EU"
    )

    gdpr_docs = parselite_folder(
        DOCS_DIR / "GDPR",
        source_type="gdpr",
        authority_rank=2,
        jurisdiction="EU"
    )

    other_docs = parselite_folder(
        DOCS_DIR / "OTHER_REGULATIONS",
        source_type="other",
        authority_rank=3,
        jurisdiction="GLOBAL"
    )

    all_docs = eu_docs + gdpr_docs + other_docs

    cleaned_docs = []

    for d in all_docs:

        text = d.text

        # enforce deterministic string normalization only
        text = text.replace("\r", " ")
        text = " ".join(text.split())   # collapses weird spacing

        cleaned_docs.append(
            Document(
                text=text,
                metadata=d.metadata
            )
        )

    return cleaned_docs


def build_index():

    documents = load_documents()

    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=150
    )

    # index = VectorStoreIndex.from_documents(
    #     documents=documents,
    #     transformations=[splitter],
    #     show_progress=True,
    # )

    # index.storage_context.persist(
    #     persist_dir=PERSIST_DIR
    # )

    vector_store = QdrantVectorStore(
        client=client, collection_name="eu_ai_chat")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
        transformations=[splitter],
    )
    print("Index persisted.")
    return index


def load_existing_index():

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="eu_ai_chat",
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )


def query_index(index):
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="source_type", value="eu_ai_act")
        ]
    )

    retriever = index.as_retriever(
        similarity_top_k=5,
        filters=filters
    )

    query_engine = RetrieverQueryEngine.from_args(retriever)

    response = query_engine.query(
        "What obligations does the EU AI Act impose on providers of high-risk AI systems?"
    )

    print("\nANSWER:\n")
    print(response)

    print("\nSOURCE NODES:\n")

    for node in response.source_nodes:

        print("TEXT:")
        print(node.node.text[:300])

        print("\nMETADATA:")
        print(node.node.metadata)

        print("\nSCORE:")
        print(node.score)

        print("\n" + "=" * 80 + "\n")


def load_all_docs():

    print(BASE_DIR)
    print(DOCS_DIR)
    print(DOCS_DIR.exists())

    if not os.path.exists(PERSIST_DIR):

        print("Building new index...")
        index = build_index()

    else:

        print("Loading existing index...")
        index = load_existing_index()

    query_index(index)


def simple_debug():
    docs = load_documents()
    inspect_raw_documents(docs)


def check_parsing():
    docs = load_documents()
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=150)
    inspect_chunks(docs, splitter)


if __name__ == "__main__":
    # load_all_docs()
    # simple_debug()
    check_parsing()
