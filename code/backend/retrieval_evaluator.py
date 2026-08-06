from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from refine import knowledge_refinement
from llm import llm
from state import RAGstate
from retrieve import retrieve
class RetrievalEvaluation(BaseModel):
    score: int = Field(description="Reliability score from 0 to 100")
    reasoning: str = Field(description="Reason for the score")
    verdict: Literal["correct", "ambiguous", "incorrect"]


structured_llm = llm.with_structured_output(RetrievalEvaluation)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a retrieval evaluator for a Retrieval-Augmented Generation (RAG) system.

Your task is to determine whether the retrieved documents are reliable enough
to answer the user's question.

Evaluate:
- Relevance to the question
- Completeness
- Accuracy
- Coverage

Assign a score between 0 and 100.

Classification:
- 80-100 → correct
- 50-79 → ambiguous
- 0-49 → incorrect

Return the score, verdict, and a short reasoning.
""",
        ),
        (
            "human",
            """
Question:
{question}

Retrieved Documents:
{documents}
""",
        ),
    ]
)

def evaluate_retrieval(question: str, documents: list):
    docs = "\n\n".join(
        [doc.page_content if hasattr(doc, "page_content") else str(doc)
         for doc in documents]
    )

    chain = prompt | structured_llm

    return chain.invoke(
        {
            "question": question,
            "documents": docs,
        }
    )

def retrieval_evaluator(state: RAGstate) -> RAGstate:
    """
    Evaluate retrieved documents using the LLM.
    """

    result = evaluate_retrieval(
        question=state["question"],
        documents=state["documents"],
    )

    state["retrieval_score"] = result.score
    state["retrieval_reasoning"] = result.reasoning
    state["retrieval_verdict"] = result.verdict

    return state

def retrieval_route(state: RAGstate) -> str:
    return state["retrieval_verdict"]


if __name__ == "__main__":
    question = input("Enter your question: ")

    # Initial state
    state: RAGstate = {
        "question": question,
        "documents": [],
    }

    # Retrieve documents
    state = retrieve(state)

    print("\nRetrieved Documents:\n")
    for i, doc in enumerate(state["documents"], start=1):
        print(f"Document {i}")
        print(doc.page_content)
        print("-" * 80)

    # Evaluate retrieval
    state = retrieval_evaluator(state)

    print("\nRetrieval Evaluation")
    print(f"Score     : {state['retrieval_score']}")
    print(f"Verdict   : {state['retrieval_verdict']}")
    print(f"Reasoning : {state['retrieval_reasoning']}")

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

    if decision != "refine":
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