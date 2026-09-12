import os
import uuid
import shutil
from typing import List, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langgraph.graph import StateGraph, START, END

# Import existing backend modules & schemas
from state import RAGstate
from reflection_result import ReflectionResult
from llm import llm
from refine import refine_knowledge, refine_and_search, rewrite_query
from generate import generate_answer
from retrieve_more import retrieve_more
from regenerate import regenerate_answer

# Initialize FastAPI App
app = FastAPI(title="Self-Correcting RAG API", version="1.0.0")

# Enable CORS for frontend Vite dev server and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Chroma Vectorstore & Embeddings Setup
# -------------------------------------------------------------
api_key = os.getenv("NVIDIA_API_KEY")
embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b",
    api_key=api_key
)

persist_dir = str(Path(__file__).parent / "chroma_db")
vectorstore = Chroma(
    collection_name="self_correcting_rag",
    persist_directory=persist_dir,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Uploads directory
uploads_dir = Path(__file__).parent / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

# In-memory document registry for UI display
doc_registry: Dict[str, Dict[str, Any]] = {}

def index_file(file_path: Path, doc_id: str, original_name: str) -> int:
    """Ingest a PDF or text file into Chroma DB."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
    else:
        loader = TextLoader(str(file_path))
        docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source_name"] = original_name

    if chunks:
        vectorstore.add_documents(chunks)

    doc_registry[doc_id] = {
        "id": doc_id,
        "name": original_name,
        "size": file_path.stat().st_size,
        "type": "application/pdf" if suffix == ".pdf" else "text/plain",
        "chunks": len(chunks)
    }
    return len(chunks)

# Pre-populate default PDF if collection is empty
def init_default_documents():
    default_pdf = Path(__file__).parent / "2401.15884v3.pdf"
    if default_pdf.exists():
        default_id = "default-self-rag-paper"
        if default_id not in doc_registry:
            doc_registry[default_id] = {
                "id": default_id,
                "name": "2401.15884v3.pdf (Self-RAG Paper)",
                "size": default_pdf.stat().st_size,
                "type": "application/pdf",
                "chunks": vectorstore._collection.count()
            }
        if vectorstore._collection.count() == 0:
            print("[SERVER] Ingesting default document 2401.15884v3.pdf into Chroma...")
            index_file(default_pdf, default_id, "2401.15884v3.pdf (Self-RAG Paper)")

init_default_documents()

# -------------------------------------------------------------
# Evaluator & Reflection Prompt Definitions
# -------------------------------------------------------------
class RetrievalEvaluation(BaseModel):
    score: int = Field(description="Reliability score from 0 to 100")
    reasoning: str = Field(description="Reason for the score")
    verdict: Literal["correct", "ambiguous", "incorrect"]

retrieval_eval_llm = llm.with_structured_output(RetrievalEvaluation)
retrieval_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a retrieval evaluator for a Retrieval-Augmented Generation (RAG) system.
Determine whether the retrieved documents are reliable enough to answer the user's question.
Evaluate:
- Relevance to the question
- Completeness
- Accuracy
- Coverage
Assign a score between 0 and 100.
Classification:
- 80-100 -> correct
- 50-79 -> ambiguous
- 0-49 -> incorrect
Return score, verdict, and reasoning.""",
        ),
        (
            "human",
            "Question:\n{question}\n\nRetrieved Documents:\n{documents}",
        ),
    ]
)
retrieval_eval_chain = retrieval_eval_prompt | retrieval_eval_llm

def server_retrieve(state: RAGstate) -> RAGstate:
    question = state["question"]
    docs = retriever.invoke(question)
    state["documents"] = docs
    return state

def server_retrieval_evaluator(state: RAGstate) -> RAGstate:
    question = state["question"]
    documents = state.get("documents", [])
    docs_text = "\n\n".join(
        [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in documents]
    )
    if not docs_text.strip():
        state["retrieval_score"] = 0
        state["retrieval_reasoning"] = "No relevant documents found in knowledge base."
        state["retrieval_verdict"] = "incorrect"
        return state

    result = retrieval_eval_chain.invoke({"question": question, "documents": docs_text})
    state["retrieval_score"] = result.score
    state["retrieval_reasoning"] = result.reasoning
    state["retrieval_verdict"] = result.verdict
    return state

def server_retrieval_route(state: RAGstate) -> str:
    verdict = state.get("retrieval_verdict", "incorrect")
    return verdict

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a RAG answer evaluator.
Evaluate the answer using ONLY the provided context.
Check:
1. Is the answer grounded in the context?
2. Does it answer all parts of the question?
3. Does it contain hallucinations?
4. Is additional retrieval needed?
Verdict rules:
- good: Answer is grounded and complete.
- regenerate: Context appears sufficient but answer quality is poor, incomplete, or poorly written.
- retrieve_more: Context lacks information needed to answer confidently.""",
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}",
        ),
    ]
)
reflection_chain = reflection_prompt | llm.with_structured_output(ReflectionResult)

def server_self_reflection(state: RAGstate) -> RAGstate:
    question = state["question"]
    answer = state.get("answer", "")

    context_docs = []
    if state.get("refined_documents"):
        context_docs.extend(state["refined_documents"])
    if state.get("web_docs"):
        context_docs.extend(state["web_docs"])
    if not context_docs and state.get("documents"):
        context_docs.extend(state["documents"])

    context = "\n\n".join(doc.page_content for doc in context_docs)

    reflection = reflection_chain.invoke(
        {
            "question": question,
            "context": context,
            "answer": answer,
        }
    )
    state["reflection"] = reflection
    return state

def server_reflection_route(state: RAGstate) -> str:
    verdict = state["reflection"].verdict
    if verdict == "good":
        return "finish"
    elif verdict == "retrieve_more":
        return "retrieve_more"
    elif verdict == "regenerate":
        return "regenerate"
    return "finish"

# -------------------------------------------------------------
# LangGraph Workflow Definition
# -------------------------------------------------------------
builder = StateGraph(RAGstate)
builder.add_node("retrieve", server_retrieve)
builder.add_node("retrieval_evaluator", server_retrieval_evaluator)
builder.add_node("refine", refine_knowledge)
builder.add_node("extra", refine_and_search)
builder.add_node("rewrite", rewrite_query)
builder.add_node("generate", generate_answer)
builder.add_node("reflect", server_self_reflection)
builder.add_node("retrieve_more", retrieve_more)
builder.add_node("regenerate", regenerate_answer)

# Graph Edges
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "retrieval_evaluator")

builder.add_conditional_edges(
    "retrieval_evaluator",
    server_retrieval_route,
    {
        "correct": "refine",
        "ambiguous": "extra",
        "incorrect": "rewrite",
    }
)

builder.add_edge("refine", "generate")
builder.add_edge("extra", "generate")
builder.add_edge("rewrite", "generate")

builder.add_edge("generate", "reflect")

builder.add_conditional_edges(
    "reflect",
    server_reflection_route,
    {
        "finish": END,
        "retrieve_more": "retrieve_more",
        "regenerate": "regenerate",
    }
)

builder.add_edge("retrieve_more", "regenerate")
builder.add_edge("regenerate", END)

rag_graph = builder.compile()

# -------------------------------------------------------------
# API Request / Response Schemas
# -------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    name: str
    type: str = "document"
    page: Optional[int] = None
    url: Optional[str] = None
    snippet: Optional[str] = None

class EvaluationData(BaseModel):
    score: int
    verdict: str
    reasoning: str

class ReflectionData(BaseModel):
    verdict: str
    grounded: bool
    hallucination: bool
    confidence: int
    reasoning: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    evaluation: EvaluationData
    reflection: ReflectionData
    steps: List[str]

# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "total_embeddings": vectorstore._collection.count(),
        "total_documents": len(doc_registry)
    }

@app.get("/api/documents")
def list_documents():
    return list(doc_registry.values())

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    uploaded = []
    for file in files:
        doc_id = str(uuid.uuid4())
        dest = uploads_dir / f"{doc_id}_{file.filename}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        chunks_count = index_file(dest, doc_id, file.filename)
        uploaded.append(doc_registry[doc_id])
    return {"message": f"Successfully processed {len(uploaded)} files", "documents": uploaded}

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    if doc_id not in doc_registry:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        vectorstore._collection.delete(where={"doc_id": doc_id})
    except Exception as e:
        print(f"[SERVER] Error deleting from chroma: {e}")
    del doc_registry[doc_id]
    return {"message": "Document removed successfully"}

@app.post("/api/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    steps = ["1. Initialized Self-Corrective RAG workflow."]

    initial_state = {
        "question": req.question.strip(),
        "documents": [],
        "refined_documents": [],
        "web_docs": [],
        "retrieval_score": 0,
        "retrieval_verdict": "",
        "retrieval_reasoning": "",
        "answer": "",
        "final_answer": "",
    }

    try:
        final_state = rag_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {str(e)}")

    retrieved_count = len(final_state.get("documents", []))
    score = final_state.get("retrieval_score", 0)
    retrieval_verdict = final_state.get("retrieval_verdict", "unknown")
    retrieval_reasoning = final_state.get("retrieval_reasoning", "")

    steps.append(f"2. Retrieved {retrieved_count} document chunks from Chroma DB.")
    steps.append(f"3. Retrieval Evaluator score: {score}/100 -> Route: '{retrieval_verdict}'. ({retrieval_reasoning})")

    if retrieval_verdict == "correct":
        steps.append("4. Knowledge Refinement: Filtered irrelevant sentences from retrieved passages.")
    elif retrieval_verdict == "ambiguous":
        steps.append("4. Refinement & Web Search: Augmented retrieved documents with web search queries.")
    elif retrieval_verdict == "incorrect":
        steps.append("4. Query Rewriting: Reformulated query and fetched external web context.")

    steps.append("5. Generated contextual answer using LLM.")

    refl = final_state.get("reflection")
    refl_verdict = refl.verdict if refl else "good"
    refl_grounded = refl.grounded_in_context if refl else True
    refl_hallucination = refl.hallucination_detected if refl else False
    refl_confidence = refl.confidence if refl else 100
    refl_reasoning = refl.reasoning if refl else "Answer successfully grounded."

    steps.append(f"6. Self-Reflection: Verdict '{refl_verdict}' | Grounded: {refl_grounded} | Hallucination: {refl_hallucination}. ({refl_reasoning})")

    if refl_verdict == "retrieve_more":
        steps.append("7. Triggered additional knowledge retrieval and regenerated improved answer.")
    elif refl_verdict == "regenerate":
        steps.append("7. Regenerated answer to improve completeness and structure.")
    else:
        steps.append("7. Self-correction complete. Final answer accepted.")

    final_answer = final_state.get("final_answer") or final_state.get("answer") or "No answer generated."

    # Compile unique sources
    raw_sources = []
    if final_state.get("documents"):
        raw_sources.extend(final_state["documents"])
    if final_state.get("refined_documents"):
        raw_sources.extend(final_state["refined_documents"])
    if final_state.get("web_docs"):
        raw_sources.extend(final_state["web_docs"])
    if final_state.get("web_documents"):
        raw_sources.extend(final_state["web_documents"])

    unique_sources: Dict[str, SourceItem] = {}
    for doc in raw_sources:
        meta = getattr(doc, "metadata", {}) or {}
        name = meta.get("source_name") or meta.get("title") or meta.get("source") or meta.get("url") or "Document"
        name_clean = str(Path(name).name) if not str(name).startswith("http") else str(name)
        page = meta.get("page")
        url = meta.get("url")
        doc_type = "web" if str(meta.get("source", "")).lower() == "web" or url else "document"

        key = f"{name_clean}_{page}_{url}"
        if key not in unique_sources:
            snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            unique_sources[key] = SourceItem(
                name=name_clean,
                type=doc_type,
                page=int(page) if page is not None else None,
                url=url,
                snippet=snippet
            )

    return QueryResponse(
        answer=final_answer,
        sources=list(unique_sources.values())[:6],
        evaluation=EvaluationData(
            score=score,
            verdict=retrieval_verdict,
            reasoning=retrieval_reasoning
        ),
        reflection=ReflectionData(
            verdict=refl_verdict,
            grounded=refl_grounded,
            hallucination=refl_hallucination,
            confidence=refl_confidence,
            reasoning=refl_reasoning
        ),
        steps=steps
    )

# Mount frontend/dist for unified full-stack serving
from fastapi.staticfiles import StaticFiles
dist_path = Path(__file__).parent.parent / "frontend" / "dist"
if dist_path.exists():
    app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
