import unittest
from app import config


class TestConfig(unittest.TestCase):

    def test_debug_flag_exists(self):
        self.assertIn(config.DEBUG, [True, False])

    def test_logger_exists(self):
        self.assertIsNotNone(config.logger)


if __name__ == "__main__":
    unittest.main()
