import unittest
from unittest.mock import patch, MagicMock

from app.rag import get_retriever


class TestRAG(unittest.TestCase):

    @patch("app.rag.load_vectorstore")
    def test_get_retriever_success(self, mock_load_vs):
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = "retriever"

        mock_load_vs.return_value = mock_vs

        retriever = get_retriever()
        self.assertEqual(retriever, "retriever")

    @patch("app.rag.load_vectorstore")
    def test_get_retriever_failure(self, mock_load_vs):
        mock_load_vs.side_effect = Exception("fail")

        with self.assertRaises(Exception):
            get_retriever()


if __name__ == "__main__":
    unittest.main()
