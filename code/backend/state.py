from typing import TypedDict, List

from langchain_core.documents import Document

class RAGstate(TypedDict):
    question: str
    documents: List[Document]
    refined_documents: List[Document]
    retrieval_score: int
    retrieval_verdict: str
    retrieval_reasoning: str