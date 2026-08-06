from typing import TypedDict, List
from langgraph.graph import StateGraph,START , END

from langchain_community.document_loaders import PyPDFLoader

from .state import RAGstate

loader = PyPDFLoader("data/policy.pdf")
docs = loader.load()

from user_query import user_query
from retrieve import retrieve
from retrieval_evaluator import retrieval_evaluator, retrieval_route
from refine import refine_knowledge, refine_and_search, rewrite_query

builder = StateGraph(RAGstate)
builder.add_node("user_query",user_query)
builder.add_node("retrieve", retrieve)
builder.add_node("retrieval_evaluator", retrieval_evaluator)
builder.add_node("refine", refine_knowledge)
builder.add_node("extra", refine_and_search)
builder.add_node("rewrite", rewrite_query)
builder.add_node("generate", generate_answer)
builder.add_node("reflect", self_reflection)
builder.add_node("retrieve_more", retrieve_more)
builder.add_node("regenerate", regenerate_answer)

# Entry
builder.add_edge(START, "retrieve")
# Retrieval
builder.add_edge("retrieve", "retrieval_evaluator")

builder.add_conditional_edges(
    "retrieval_evaluator",
    retrieval_route,
    {
        "refine": "refine",
        "extra": "extra",
        "rewrite": "rewrite",
    }
)

# All paths merge

builder.add_edge("refine", "generate")
builder.add_edge("extra", "generate")
builder.add_edge("rewrite", "generate")

# Reflection

builder.add_edge("generate", "reflect")

builder.add_conditional_edges(
    "reflect",
    reflection_route,
    {
        "finish": END,
        "retrieve_more": "retrieve_more",
        "regenerate": "regenerate",
    }
)

# Retry loops

builder.add_edge("retrieve_more", "generate")
builder.add_edge("regenerate", "reflect")

graph = builder.compile()