# ingest.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
print("Loading PDF...")

loader = PyPDFLoader("documents/Resume_SP_27.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

print("Splitting into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

print("Creating embeddings...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Creating Vector DB...")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Vector database created successfully!")