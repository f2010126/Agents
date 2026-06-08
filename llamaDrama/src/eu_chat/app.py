# Just a simple retrieval
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.query_engine import RetrieverQueryEngine

from qdrant_client import QdrantClient
from llamaDrama.src.eu_chat.init_llms import init_models
from llamaDrama.src.eu_chat.constants import COLLECTION

init_models()
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
