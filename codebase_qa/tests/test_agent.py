import unittest
from unittest.mock import patch, MagicMock

from app.agent import build_agent


class TestAgent(unittest.TestCase):

    @patch("app.agent.ChatGoogleGenerativeAI")
    @patch("app.agent.create_react_agent")
    def test_build_agent_success(self, mock_create_agent, mock_llm):
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        agent = build_agent()

        self.assertIsNotNone(agent)

    @patch("app.agent.ChatGoogleGenerativeAI")
    def test_build_agent_failure(self, mock_llm):
        mock_llm.side_effect = Exception("LLM fail")

        with self.assertRaises(Exception):
            build_agent()


if __name__ == "__main__":
    unittest.main()
