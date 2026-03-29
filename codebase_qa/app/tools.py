from app.rag import get_retriever

retriever = get_retriever()


def serach_code_base(query: str) -> str:
    docs = retriever._get_relevant_documents(query=query)

    formatted = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        formatted.append(f"FILE: {source}\n{d.page_content}")

    return "\n\n".join(formatted)
