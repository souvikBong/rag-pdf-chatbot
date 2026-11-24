from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

# Initialize once
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

llm = ChatOllama(
    model="qwen2.5:3b"
)


def ask_question(query):

    results = db.similarity_search(
        query,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    prompt = f"""
You are a document question-answering assistant.

Rules:
1. Answer ONLY from the provided context.
2. Do NOT use outside knowledge.
3. Do NOT make assumptions.
4. If the answer is not present in the context, reply:
   "The information is not available in the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": results
    }


# Terminal mode (optional)
if __name__ == "__main__":
    question = input("Ask a question: ")

    result = ask_question(question)

    print("\nAnswer:\n")
    print(result["answer"])