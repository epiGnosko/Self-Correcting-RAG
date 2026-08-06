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


def knowledge_refinement(state: RAGstate) -> RAGstate:
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

    state = knowledge_refinement(state)

    print(
        f"\nRefined to {len(state['refined_documents'])} document(s).\n"
    )

    for i, doc in enumerate(state["refined_documents"], start=1):
        print("=" * 80)
        print(f"DOCUMENT {i}")
        print("=" * 80)
        print(doc.page_content)
        print()