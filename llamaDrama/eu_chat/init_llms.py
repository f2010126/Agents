# declare and init the LLMs and models that we use in this project
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import os
from llamaDrama.eu_chat.constants import GOOGLE_API_KEY, MODEL

# also adding it to my os env.
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY
# just why is goofle different from Gemini?
os.environ["INSTRUCTOR_PROVIDER"] = "google"


def init_models():
    Settings.llm = Gemini(
        model_name=MODEL,
        api_key=GOOGLE_API_KEY
    )

    Settings.embed_model = GeminiEmbedding(
        model_name="gemini-embedding-001",
        api_key=GOOGLE_API_KEY
    )
