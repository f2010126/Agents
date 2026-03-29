import unittest
from unittest.mock import patch, MagicMock

from app.tools import search_codebase


class TestTools(unittest.TestCase):

    @patch("app.tools.retriever")
    def test_search_codebase_success(self, mock_retriever):
        mock_doc = MagicMock()
        mock_doc.page_content = "def foo(): pass"
        mock_doc.metadata = {"source": "file.py"}

        mock_retriever.get_relevant_documents.return_value = [mock_doc]

        result = search_codebase("foo")

        self.assertIn("file.py", result)
        self.assertIn("def foo()", result)

    @patch("app.tools.retriever")
    def test_search_codebase_empty(self, mock_retriever):
        mock_retriever.get_relevant_documents.return_value = []

        result = search_codebase("nothing")

        self.assertIn("No relevant code", result)

    @patch("app.tools.retriever")
    def test_search_codebase_error(self, mock_retriever):
        mock_retriever.get_relevant_documents.side_effect = Exception("fail")

        result = search_codebase("error")

        self.assertIn("Error retrieving", result)


if __name__ == "__main__":
    unittest.main()
