from typing import TypedDict, List

from langchain_core.documents import Document

class RAGstate(TypedDict):
    question: str
    documents: List[Document]
    retrieval_quality: str