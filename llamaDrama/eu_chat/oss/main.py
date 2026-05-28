# https://developers.llamaindex.ai/python/examples/low_level/oss_ingestion_retrieval/
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

info = client.get_collection("eu_ai_chat")

print(info.points_count)

points, _ = client.scroll(
    collection_name="eu_ai_chat",
    limit=3,
    with_payload=True,
    with_vectors=False
)

for p in points:
    print(p.payload)
