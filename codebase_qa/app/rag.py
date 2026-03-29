# Handles the retrival stuff


from langchain_community.vectorstores import FAISS
# FAISS (Facebook AI Similarity Search) is an open-source library by Meta
# for efficient, high-performance similarity search and clustering of dense, high-dimensional vectors.
# using it for semantic search
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def load_vectorstore(path="vectorstore"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_retriever():
    vs = load_vectorstore()
    return vs.as_retriever(search_kwargs={"k": 5})  # k most relevant
