from langchain_core.messages import HumanMessage
from state import RAGstate
from llm import llm
def regenerate_answer(state: RAGstate) -> RAGstate:
    """
    Regenerate answer when reflection says the answer
    is poorly formed but additional retrieval is not needed.
    """

    question = state["question"]
    documents = state["documents"]
    reflection = state["reflection"]
    answer = state["answer"]

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
You are improving a previous RAG answer.

Question:
{question}

Retrieved Context:
{context}

Previous Answer:
{answer}

Reflection Feedback:
{reflection.reasoning}

Instructions:
- Fix the issues mentioned in the feedback.
- Use only the provided context.
- Do not hallucinate.
- Make the answer complete and well-structured.
- If information is unavailable in the context, say so.
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    state["final_answer"] = response.content

    return state

if __name__ == "__main__":

    class MockReflection:
        reasoning = (
            "The answer is incomplete. It mentions LangGraph but does not "
            "explain how conditional edges work."
        )

    from langchain_core.documents import Document

    test_state = {
        "question": "What is LangGraph and how do conditional edges work?",
        "documents": [
            Document(
                page_content="""
                LangGraph is a framework for building stateful AI workflows.
                Conditional edges allow routing to different nodes based on
                the output of a routing function.
                """
            )
        ],
        "reflection": MockReflection(),
        "answer": "LangGraph is a framework for building stateful AI workflows."
    }

    try:
        updated_state = regenerate_answer(test_state)

        print("\n" + "=" * 80)
        print("REGENERATED ANSWER")
        print("=" * 80)
        print(updated_state["final_answer"])
        print("=" * 80)

    except Exception as e:
        print(f"Error during regeneration: {e}")