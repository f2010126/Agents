# Handles the retrival stuff
from langchain_community.vectorstores import FAISS
# FAISS (Facebook AI Similarity Search) is an open-source library by Meta
# for efficient, high-performance similarity search and clustering of dense, high-dimensional vectors.
# using it for semantic search
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import logger


def load_vectorstore(path="vectorstore"):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

        vs = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        logger.info("Vectorstore loaded successfully")
        return vs

    except Exception as e:
        logger.error(f"Failed to load vectorstore: {e}")
        raise


def get_retriever():
    try:
        vs = load_vectorstore()
        return vs.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        logger.error(f"Retriever creation failed: {e}")
        raise
