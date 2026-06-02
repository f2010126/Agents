# Tools used by the Agents
# 1. Retrieve what sources are available in vector DB.
# 2. Websearch
# 3. Retrieve from vector DB
from qdrant_client import QdrantClient
from constants import MODEL, GOOGLE_API_KEY, COLLECTION

client = QdrantClient(host="localhost", port=6333)


def list_available_documents():
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


if __name__ == "__main__":
    print(list_available_documents())
