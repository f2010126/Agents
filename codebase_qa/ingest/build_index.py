from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


loader = DirectoryLoader("../data", glob="**/*.py")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

db = FAISS.from_documents(chunks, embeddings)
db.save_local("../vectorstore")
