from reflection_result import ReflectionResult
from state import RAGstate
from llm import llm

from langchain_core.prompts import ChatPromptTemplate

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a RAG answer evaluator.

Evaluate the answer using ONLY the provided context.

Check:

1. Is the answer grounded in the context?
2. Does it answer all parts of the question?
3. Does it contain hallucinations?
4. Is additional retrieval needed?

Verdict rules:

- good
  Answer is grounded and complete.

- regenerate
  Context appears sufficient but answer quality is poor,
  incomplete, or poorly written.

- retrieve_more
  Context lacks information needed to answer confidently.
""",
        ),
        (
            "human",
            """
Question:
{question}

Context:
{context}

Answer:
{answer}
"""
        ),
    ]
)
reflection_chain = (
    reflection_prompt
    | llm.with_structured_output(ReflectionResult)
)
def self_reflection(state: RAGstate) -> RAGstate:

    question = state["question"]
    answer = state["answer"]

    context_docs = []

    if state.get("refined_documents"):
        context_docs.extend(state["refined_documents"])

    if state.get("web_docs"):
        context_docs.extend(state["web_docs"])

    if not context_docs:
        context_docs.extend(state["documents"])

    context = "\n\n".join(
        doc.page_content
        for doc in context_docs
    )

    reflection = reflection_chain.invoke(
        {
            "question": question,
            "context": context,
            "answer": answer,
        }
    )

    state["reflection"] = reflection

    return state
def reflection_route(state: RAGstate) -> str:
    """
    Route based on reflection verdict.

    Returns:
        finish
        retrieve_more
        regenerate
    """

    verdict = state["reflection"].verdict

    if verdict == "good":
        return "finish"

    elif verdict == "retrieve_more":
        return "retrieve_more"

    elif verdict == "regenerate":
        return "regenerate"

    else:
        raise ValueError(
            f"Unknown reflection verdict: {verdict}"
        )

if __name__ == "__main__":

    from langchain_core.documents import Document

    # -------------------------
    # Dummy State
    # -------------------------

    state = {
        "question": "Do we need NumPy to implement RAG?",

        "documents": [],

        "refined_documents": [
            Document(
                page_content="""
A Retrieval-Augmented Generation (RAG) system can be
implemented using embeddings, a vector database,
and a language model.

NumPy is not a required component of a RAG system.
"""
            )
        ],

        "web_docs": [],

        "retrieval_score": 0,
        "retrieval_verdict": "",
        "retrieval_reasoning": "",

        "answer": """
No. NumPy is not required to implement a RAG system.
A RAG pipeline primarily needs embeddings, retrieval,
and a language model.
""",
    }

    print("\n==============================")
    print("RUNNING SELF REFLECTION")
    print("==============================")

    state = self_reflection(state)

    reflection = state["reflection"]

    print("\n==============================")
    print("REFLECTION RESULT")
    print("==============================")

    print(reflection.model_dump())

    print("\n==============================")
    print("ROUTE")
    print("==============================")

    route = reflection_route(state)

    print(route)