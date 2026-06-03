# Tools used by the Agents
# 1. Retrieve what sources are available in vector DB.
# 2. Websearch
# 3. Retrieve from vector DB
from qdrant_client import QdrantClient
from constants import MODEL, GOOGLE_API_KEY, COLLECTION
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field
client = QdrantClient(host="localhost", port=6333)


vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION,
    enable_hybrid=True
)

index = VectorStoreIndex.from_vector_store(vector_store)
# Available sources


@tool("List Available Documents")
def list_doc_tool():
    """
    Lists available knowledge sources and jurisdictions in the vector database.
    Use before retrieval when determining whether relevant sources exist.
    """
    scroll_result, _ = client.scroll(
        collection_name=COLLECTION,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )

    seen = {}

    for point in scroll_result:
        p = point.payload

        name = p.get("file_name")
        jurisdiction = p.get("jurisdiction_label")

        if not name or not jurisdiction:
            continue

        key = (name, jurisdiction)

        if key not in seen:
            seen[key] = {
                "filename": name,
                "jurisdiction": jurisdiction
            }

    return list(seen.values())

# Retrieval
# add metadata filters based on bias.


def build_filters(mode: str):
    """
    metadata biasing only. No hard exclusion unless STRICT
    """
    # match exact (edge case)
    if mode == "strict":
        return MetadataFilters(
            filters=[
                ExactMatchFilter(
                    key="source_type",
                    value="eu_ai_act"
                )
            ]
        )

    if mode == "focused":
        # soft bias handled outside Qdrant (No hard filter)
        return None

    return None  # no metadata filtering here

# actual query to Qdrant


def retrieve(query: str, mode: str = "broad", top_k: int = 10):
    """
    call to the Qdrant collection
    """
    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=build_filters(mode)
    )

    nodes = retriever.retrieve(query)

    results = []

    for n in nodes:
        results.append({
            "text": n.node.text,
            "metadata": n.node.metadata,
            "score": getattr(n, "score", None)
        })

    return results

# small weighted sorting after retrieval. No reranking as such


def rerank_bias(results):
    """
    scoring adjustment (not model rerank).
    """

    def score(item):
        meta = item["metadata"]

        boost = 0

        if meta.get("source_type") == "eu_ai_act":
            boost += 0.3

        if meta.get("authority_rank") == 1:
            boost += 0.2

        return boost

    return sorted(
        results,
        key=lambda x: (x.get("score") or 0) + score(x),
        reverse=True
    )

# put it together as a tool


class RetrievalInput(BaseModel):
    query: str = Field(
        description="The user's search question"
    )

    mode: str = Field(
        default="broad",
        description="broad, focused, or strict"
    )

    top_k: int = Field(
        default=10,
        description="Number of results to retrieve"
    )


class RetrievalTool(BaseTool):
    name: str = "Knowledge Retrieval"

    description: str = """
    Searches the legal knowledge base stored in Qdrant.

    Use when the answer is likely in internal documents.

    Modes:
    - broad: general search across all documents
    - focused: prioritizes EU AI Act and authoritative sources
    - strict: only EU AI Act content
    """

    args_schema = RetrievalInput

    def _run(self, query: str, mode: str = "broad", top_k: int = 10):
        return retrieval_tool(query=query, mode=mode, top_k=top_k)


def retrieval_tool(query: str, mode: str = "broad", top_k: int = 10):
    """
    Searches the legal knowledge base stored in Qdrant.

    Use this tool whenever the answer may be contained in uploaded legal documents.

    Modes:
    - broad: search all documents
    - focused: prioritize EU AI Act and authoritative sources
    - strict: search only EU AI Act content
    """

    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=build_filters(mode)
    )

    nodes = retriever.retrieve(query)

    results = [
        {
            "text": n.node.text,
            "metadata": n.node.metadata,
            "score": getattr(n, "score", None)
        }
        for n in nodes
    ]

    # only focused mode applies biasing
    if mode == "focused":
        results = rerank_bias(results)

    return results


if __name__ == "__main__":
    print(list_doc_tool())
