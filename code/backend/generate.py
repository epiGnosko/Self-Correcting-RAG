from langchain_core.prompts import ChatPromptTemplate
from state import RAGstate
from llm import llm


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a RAG answer generation assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Use only information present in the context.
- Do not use outside knowledge.
- If the answer cannot be found in the context, say:

"I could not find enough information in the retrieved documents."

- Be concise and accurate.
""",
        ),
        (
            "human",
            """
Question:
{question}

Context:
{context}
""",
        ),
    ]
)

answer_chain = answer_prompt | llm


def generate_answer(state: RAGstate) -> RAGstate:
    """
    Generate final answer from retrieved/refined documents.
    """

    question = state["question"]

    # --------------------------------
    # Build context
    # --------------------------------

    context_docs = []

    if state.get("refined_documents"):
        context_docs.extend(state["refined_documents"])

    if state.get("web_docs"):
        context_docs.extend(state["web_docs"])

    # fallback
    if not context_docs:
        context_docs.extend(state["documents"])

    context = "\n\n".join(
        doc.page_content
        for doc in context_docs
    )

    # --------------------------------
    # Generate answer
    # --------------------------------

    response = answer_chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    answer = response.content.strip()

    state["answer"] = answer

    return state

if __name__ == "__main__":
    from refine import (
        refine_knowledge,
        refine_and_search,
        rewrite_query,
    )
    from retrieve import retrieve
    from retrieval_evaluator import (
        retrieval_evaluator,
        retrieval_route,
    )

    question = input("Enter your question: ")

    state = {
        "question": question,
        "documents": [],
        "refined_documents": [],
        "web_docs": [],
        "retrieval_score": 0,
        "retrieval_verdict": "",
        "retrieval_reasoning": "",
        "answer": "",
    }

    print("\n==============================")
    print("Running Retrieval...")
    print("==============================")

    state = retrieve(state)

    print(
        f"\nRetrieved {len(state['documents'])} documents."
    )

    print("\n==============================")
    print("Running Retrieval Evaluator...")
    print("==============================")

    state = retrieval_evaluator(state)

    decision = retrieval_route(state)

    print(f"\nDecision: {decision}")

    # -------------------------
    # Route handling
    # -------------------------

    if decision == "correct":

        print("\nRunning Knowledge Refinement...")
        state = refine_knowledge(state)

    elif decision == "ambiguous":

        print("\nRunning Refinement + Web Search...")
        state = refine_and_search(state)

    elif decision == "incorrect":

        print("\nRunning Query Rewrite + Web Search...")
        state = rewrite_query(state)

    else:
        raise ValueError(
            f"Unknown decision: {decision}"
        )

    # -------------------------
    # Generate Answer
    # -------------------------

    print("\n==============================")
    print("Generating Answer...")
    print("==============================")

    state = generate_answer(state)

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")

    print(state["answer"])