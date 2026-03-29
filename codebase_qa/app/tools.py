from app.rag import get_retriever
from app.config import logger

retriever = get_retriever()


def search_codebase(query: str) -> str:
    logger.debug(f"[TOOL] search_codebase called with query: {query}")

    try:
        docs = retriever.get_relevant_documents(query)

        if not docs:
            logger.warning("No documents retrieved")
            return "No relevant code found."

        formatted = []
        for d in docs:
            source = d.metadata.get("source", "unknown")
            formatted.append(f"FILE: {source}\n{d.page_content}")

        result = "\n\n".join(formatted)

        logger.debug(f"[TOOL] Retrieved {len(docs)} documents")
        return result

    except Exception as e:
        logger.error(f"[TOOL ERROR] search_codebase failed: {e}")
        return "Error retrieving codebase."
