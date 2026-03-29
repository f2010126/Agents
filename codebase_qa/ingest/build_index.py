from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    loader = DirectoryLoader("../data", glob="**/*.py")
    docs = loader.load()

    logger.info(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("../vectorstore")

    logger.info("Vectorstore built and saved successfully")

except Exception as e:
    logger.error(f"Indexing failed: {e}")
