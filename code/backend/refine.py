from langchain_core.documents import Document
from state import RAGstate
from typing import List
from llm import llm
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from state import RAGstate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a document refinement assistant.

Your task is to remove all information that is NOT required to answer the user's question.

Rules:

- Read the document carefully.
- Keep only the sentences that help answer the question.
- Remove unrelated explanations, examples and background.
- Do NOT summarize.
- Do NOT rewrite.
- Do NOT change wording.
- Simply return the useful sentences exactly as they appear.
- If nothing is useful, return exactly:

EMPTY
""",
        ),
        (
            "human",
            """
Question:
{question}

Document:
{document}
""",
        ),
    ]
)

chain = prompt | llm


def refine_knowledge(state: RAGstate) -> RAGstate:
    """
    Filters each retrieved document independently using the LLM.
    One LLM call is made per document.
    """

    question = state["question"]
    documents = state["documents"]

    refined_docs: List[Document] = []

    for doc in documents:

        response = chain.invoke(
            {
                "question": question,
                "document": doc.page_content,
            }
        )

        refined_text = response.content.strip()

        if refined_text.upper() == "EMPTY":
            continue

        refined_docs.append(
            Document(
                page_content=refined_text,
                metadata=doc.metadata,
            )
        )

    state["refined_documents"] = refined_docs

    return state


from search_tool import SearchTool,web_search

# ----------------------------
# Prompt to detect missing info
# ----------------------------

missing_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert retrieval evaluator.

You are given:
1. The user's question.
2. The refined retrieved documents.

Determine whether the refined documents completely answer the user's question.

Rules:
- If they completely answer the question, return exactly:

NONE

- Otherwise, return ONLY a concise search query that would retrieve the missing information.
- Do not explain your reasoning.
- Do not answer the question.
- Return only the search query.
""",
        ),
        (
            "human",
            """
Question:
{question}

Refined Documents:
{documents}
""",
        ),
    ]
)

missing_chain = missing_prompt | llm


def refine_and_search(state: RAGstate) -> RAGstate:
    """
    Refine retrieved documents and supplement them
    with web search results when information is missing.
    """

    # -----------------------------------
    # Step 1: Refine retrieved documents
    # -----------------------------------

    state = refine_knowledge(state)

    question = state["question"]
    refined_docs = state["refined_documents"]

    # -----------------------------------
    # Step 2: Combine refined documents
    # -----------------------------------

    combined_docs = "\n\n".join(
        doc.page_content for doc in refined_docs
    )

    # -----------------------------------
    # Step 3: Detect missing information
    # -----------------------------------

    response = missing_chain.invoke(
        {
            "question": question,
            "documents": combined_docs,
        }
    )

    search_query = response.content.strip()

    print(f"Missing info query: {search_query}")

    # -----------------------------------
    # Step 4: Search the web (if needed)
    # -----------------------------------

    web_docs = []

    if search_query.upper() != "NONE":

        print("Searching the web...")

        web_docs = web_search.invoke(
            {"query": search_query}
        )

    # -----------------------------------
    # Step 5: Merge documents
    # -----------------------------------

    all_docs = refined_docs + web_docs

    state["web_documents"] = web_docs
    state["documents"] = all_docs

    return state

if __name__ == "__main__":

    from retrieve import retrieve
    from retrieval_evaluator import (
        retrieval_evaluator,
        retrieval_route,
    )

    # ---------------------------
    # User Question
    # ---------------------------
    question = input("Enter your question: ")

    state = {
        "question": question,
        "documents": [],
        "refined_documents": [],
    }

    print("\n==============================")
    print("Running Retrieval...")
    print("==============================")

    state = retrieve(state)

    print(f"\nRetrieved {len(state['documents'])} documents.\n")

    print("==============================")
    print("Running Retrieval Evaluator...")
    print("==============================")

    state = retrieval_evaluator(state)

    decision = retrieval_route(state)

    print(f"\nDecision: {decision}")

    if decision != "correct":
        print("\nRetrieval evaluator rejected the documents.")
        exit()

    print("\n==============================")
    print("Running Knowledge Refinement...")
    print("==============================")

    state = refine_knowledge(state)

    print(
        f"\nRefined to {len(state['refined_documents'])} document(s).\n"
    )

    for i, doc in enumerate(state["refined_documents"], start=1):
        print("=" * 80)
        print(f"DOCUMENT {i}")
        print("=" * 80)
        print(doc.page_content)
        print()