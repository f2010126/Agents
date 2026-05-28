from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.query_engine import RetrieverQueryEngine

from qdrant_client import QdrantClient
from oss.constants import MODEL, GOOGLE_API_KEY, COLLECTION

Settings.llm = Gemini(
    model_name=MODEL,
    api_key=GOOGLE_API_KEY
)

Settings.embed_model = GeminiEmbedding(
    model_name="gemini-embedding-001",
    api_key=GOOGLE_API_KEY
)

client = QdrantClient(host="localhost", port=6333)


def load_index():

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
    )

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )


def query(index):

    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="source_type",
                value="eu_ai_act"
            )
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

    print(response)

    for node in response.source_nodes:
        print("\n---")
        print(node.node.metadata)
        print(node.node.text[:250])


if __name__ == "__main__":
    index = load_index()
    query(index)
