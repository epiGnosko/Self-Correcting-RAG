from langchain_core.documents import Document
from tavily import TavilyClient
from typing import Dict, Any
import os
from dotenv import load_dotenv
load_dotenv()

from state import RAGstate
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def retrieve_more(state: RAGstate) -> RAGstate:
    """
    Called when reflection verdict == 'retrieve_more'.
    Uses reflection reasoning to identify missing knowledge
    and retrieves additional context from the web.
    """

    reflection = state["reflection"]

    # ReflectionResult.reasoning should describe what's missing
    missing_info = reflection.reasoning

    print(f"[RETRIEVE_MORE] Missing info: {missing_info}")

    search_results = tavily.search(
        query=missing_info,
        search_depth="advanced",
        max_results=5
    )

    new_docs = []

    for result in search_results.get("results", []):
        doc = Document(
            page_content=result["content"],
            metadata={
                "source": result.get("url"),
                "title": result.get("title", ""),
                "retrieval_type": "web_search"
            }
        )
        new_docs.append(doc)

    existing_docs = state.get("documents", [])

    state["documents"] = existing_docs + new_docs

    return state

if __name__ == "__main__":

    class MockReflection:
        reasoning = "Latest developments in Retrieval-Augmented Generation (RAG)"

    test_state = {
        "reflection": MockReflection(),
        "documents": []
    }

    try:
        updated_state = retrieve_more(test_state)

        docs = updated_state["documents"]

        print(f"\nRetrieved {len(docs)} documents\n")

        for i, doc in enumerate(docs, start=1):
            print("=" * 80)
            print(f"Document {i}")
            print(f"Title: {doc.metadata.get('title')}")
            print(f"Source: {doc.metadata.get('source')}")
            print(f"Type: {doc.metadata.get('retrieval_type')}")
            print("\nContent Preview:")
            print(doc.page_content[:300])
            print("\n")

    except Exception as e:
        print(f"Error: {e}")