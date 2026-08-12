from state import RAGstate
def user_query(state: RAGstate) -> RAGstate:
    """
    Process the user's query and update the state accordingly.
    """
    question = input("Ask a question: ")
    state["question"] = question
    return state