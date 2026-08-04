from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
from state import RAGstate
from dotenv import load_dotenv
load_dotenv()
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=os.getenv("NVIDIA_API_KEY")
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


def retrieve(state: RAGstate) -> RAGstate:
    """
    Retrieve relevant documents.
    """

    question = state["question"]

    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} documents for the question: {question}")
    state["documents"] = docs

    return state


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    print(os.getenv("NVIDIA_API_KEY"))
    # Create a dummy state for testing
    state = {
        "question": "What is Self-Correcting RAG?",
        "documents": []
    }

    result = retrieve(state)

    print("\nRetrieved Documents:\n")

    for i, doc in enumerate(result["documents"], start=1):
        print(f"Document {i}")
        print("-" * 60)
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Page: {doc.metadata.get('page')}")
        print(doc.page_content[:500])  # Print first 500 characters
        print("\n")