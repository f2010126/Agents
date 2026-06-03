import unittest
from unittest.mock import patch, MagicMock

from oss.tools import list_doc_tool, retrieval_tool


@patch("oss.tools.client")
def test_groups_by_filename_and_jurisdiction(self, mock_client):
    mock_client.scroll.return_value = (
        [
            MagicMock(payload={
                "file_name": "doc1",
                "jurisdiction_label": "EU"
            }),
            MagicMock(payload={
                "file_name": "doc1",
                "jurisdiction_label": "EU"
            }),
            MagicMock(payload={
                "file_name": "doc2",
                "jurisdiction_label": "GDPR"
            }),
        ],
        None
    )
    result = list_doc_tool()

    self.assertEqual(len(result), 2)

    self.assertIn(
        {"filename": "doc1", "jurisdiction": "EU"},
        result
    )

    self.assertIn(
        {"filename": "doc2", "jurisdiction": "GDPR"},
        result
    )


class FakeNode:
    def __init__(self, text, source_type, score=0.5):
        self.node = MagicMock()
        self.node.text = text
        self.node.metadata = {"source_type": source_type}
        self.score = score


class TestRetrievalTool(unittest.TestCase):

    @patch("oss.tools.VectorStoreIndex")
    @patch("oss.tools.QdrantVectorStore")
    def test_broad_mode_returns_results(self, mock_qdrant, mock_index):

        fake_nodes = [
            FakeNode("EU AI Act text", "eu_ai_act", 0.4),
            FakeNode("GDPR text", "gdpr", 0.6),
        ]

        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = fake_nodes

        mock_index.from_vector_store.return_value.as_retriever.return_value = fake_retriever

        result = retrieval_tool(
            query="AI hiring system rules",
            mode="broad",
            top_k=5
        )

        self.assertEqual(len(result), 2)
        self.assertIn("text", result[0])
        self.assertIn("metadata", result[0])

    @patch("oss.tools.VectorStoreIndex")
    @patch("oss.tools.QdrantVectorStore")
    def test_strict_mode_applies_filter(self, mock_qdrant, mock_index):

        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = [
            FakeNode("EU AI Act only", "eu_ai_act")
        ]

        mock_index.from_vector_store.return_value.as_retriever.return_value = fake_retriever

        result = retrieval_tool(
            query="test query",
            mode="strict"
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["metadata"]["source_type"], "eu_ai_act")

    @patch("oss.tools.VectorStoreIndex")
    @patch("oss.tools.QdrantVectorStore")
    def test_focused_mode_biases_results(self, mock_qdrant, mock_index):

        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = [
            FakeNode("GDPR text", "gdpr", 0.3),
            FakeNode("EU AI Act text", "eu_ai_act", 0.2),
        ]

        mock_index.from_vector_store.return_value.as_retriever.return_value = fake_retriever

        result = retrieval_tool(
            query="data governance",
            mode="focused"
        )

        # EU AI Act should float upward after biasing
        self.assertEqual(result[0]["metadata"]["source_type"], "eu_ai_act")
