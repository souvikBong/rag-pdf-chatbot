from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

query = "What are Souvik's technical skills?"

results = db.similarity_search(
    query,
    k=3
)

print("\nRetrieved Chunks:\n")

for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)