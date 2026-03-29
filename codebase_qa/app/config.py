# Load settings here. keys etc

import os
from dotenv import load_dotenv
import logging
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)
