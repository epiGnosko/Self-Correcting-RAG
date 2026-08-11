from typing import TypedDict, List

from langchain_core.documents import Document

from reflection_result import ReflectionResult

class RAGstate(TypedDict):
    question: str
    documents: List[Document]
    refined_documents: List[Document]
    web_docs: List[Document]
    retrieval_score: int
    retrieval_verdict: str
    retrieval_reasoning: str
    answer: str
    reflection: ReflectionResult