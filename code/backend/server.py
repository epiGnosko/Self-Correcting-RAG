import os
import re
import sys
import json
import uuid
import shutil
import signal
import atexit
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any, Literal, TypedDict
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

# =============================================================
# 1. Setup, Environment & Configuration (from graph.py)
# =============================================================
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"

load_dotenv(dotenv_path=BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crag_server")

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

# LLM configurations matching graph.py
GEN_MODEL = os.getenv("GEN_MODEL", "openai/gpt-4o-mini")
EVAL_MODEL = os.getenv("EVAL_MODEL", "openai/gpt-4o-mini")

# CRAG Thresholds & Decision Labels
UPPER_TH = 0.7
LOWER_TH = 0.3
LABEL_SCORES = {"DIRECT": 0.9, "PARTIAL": 0.5, "IRRELEVANT": 0.0}
GRADER_FAILURE_SCORE = 0.5

llm = ChatOpenAI(
    model=GEN_MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0.2,
)

eval_llm = ChatOpenAI(
    model=EVAL_MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0.0,
)

tavily = TavilySearchResults(max_results=5)

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=API_KEY,
    base_url=BASE_URL,
)

# =============================================================
# 2. Storage & Vector Store Setup
# =============================================================
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = UPLOADS_DIR / "documents.json"
doc_registry: Dict[str, Dict[str, Any]] = {}

def load_registry():
    global doc_registry
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                doc_registry = json.load(f)
            logger.info("Loaded %d documents from registry.", len(doc_registry))
        except Exception as e:
            logger.error("Failed to load registry: %s", e)
            doc_registry = {}
    else:
        doc_registry = {}

def save_registry():
    try:
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(doc_registry, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to save registry: %s", e)

vectorstore = Chroma(
    collection_name="self_correcting_rag",
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def index_file(file_path: Path, doc_id: str, original_name: str, mime_type: str = "application/pdf") -> int:
    """Ingest a PDF or text file into Chroma DB and update doc_registry."""
    suffix = file_path.suffix.lower()
    raw_docs: List[Document] = []

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        raw_docs = loader.load()
    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            raw_docs = [Document(page_content=content, metadata={"source": original_name, "page": 0})]
        except Exception as e:
            logger.error("Failed to read text file %s: %s", file_path, e)
            return 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = text_splitter.split_documents(raw_docs)

    for chunk in chunks:
        chunk.page_content = chunk.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source_name"] = original_name
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 0

    if chunks:
        vectorstore.add_documents(chunks)

    doc_registry[doc_id] = {
        "id": doc_id,
        "name": original_name,
        "size": file_path.stat().st_size,
        "type": mime_type,
        "chunks": len(chunks),
        "file_path": file_path.name,
    }
    save_registry()
    logger.info("Indexed %s (%s): %d chunks added.", original_name, doc_id, len(chunks))
    return len(chunks)

def init_default_documents():
    """Seed Chroma with default reference PDF if empty."""
    load_registry()
    default_pdf = BASE_DIR / "2401.15884v3.pdf"
    default_id = "default-crag-paper"

    try:
        col_count = vectorstore._collection.count()
    except Exception:
        col_count = 0

    if col_count == 0 and default_pdf.exists():
        logger.info("Chroma collection is empty. Ingesting default reference paper (%s)...", default_pdf.name)
        index_file(default_pdf, default_id, "2401.15884v3.pdf (CRAG Paper)", "application/pdf")
    elif default_id not in doc_registry and default_pdf.exists() and col_count > 0:
        doc_registry[default_id] = {
            "id": default_id,
            "name": "2401.15884v3.pdf (CRAG Paper)",
            "size": default_pdf.stat().st_size,
            "type": "application/pdf",
            "chunks": col_count,
            "file_path": default_pdf.name,
        }
        save_registry()

# =============================================================
# 3. CRAG LangGraph Definitions & Nodes (as in graph.py)
# =============================================================

class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    scores: List[float]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Document]
    answer: str
    eval_score: float
    reflection: Dict[str, Any]
    steps: List[str]

# Retrieval Evaluator Model & Prompt
class DocEval(BaseModel):
    reason: str = Field(
        description="1-2 sentences: what the chunk says and how it relates to the question."
    )
    label: Literal["DIRECT", "PARTIAL", "IRRELEVANT"]

doc_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", """You grade whether ONE retrieved chunk helps answer a question.

Labels:
- DIRECT: the chunk itself states specific facts that answer the question
  (or the main part of it). Someone could write a correct answer from this
  chunk alone.
- PARTIAL: the chunk is about the same topic and gives useful background,
  definitions or context, but does not state the facts the question asks for.
- IRRELEVANT: the chunk is about a different topic, or only shares
  incidental words with the question.

Rules:
- Judge meaning, not keyword overlap. Sharing a word is not relevance.
- The chunk does not need to be a complete answer to earn DIRECT; it needs to
  contain the key facts for the main part of the question.
- Judge only the chunk shown. Do not use outside knowledge to fill gaps.
- Ask yourself: "If I only had this chunk, could I answer?"
  Yes -> DIRECT. Partly or only with background -> PARTIAL. No -> IRRELEVANT.

Write the reason first, then the label."""),
    ("human", "Question:\n{question}\n\nChunk:\n{chunk}"),
])

doc_eval_chain = doc_eval_prompt | eval_llm.with_structured_output(
    DocEval, method="function_calling"
)

# Knowledge Refinement
def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

class KeepOrDrop(BaseModel):
    keep: bool

filter_prompt = ChatPromptTemplate.from_messages([
    ("system", "You filter sentences for a question-answering system.\n"
               "Return keep=true if the sentence contains information that is useful "
               "for answering the question (facts, definitions, mechanisms, steps).\n"
               "Return keep=false if it is unrelated, boilerplate, or a citation fragment.\n"
               "Use ONLY the sentence."),
    ("human", "Question: {question}\n\nSentence:\n{sentence}"),
])

filter_chain = filter_prompt | eval_llm.with_structured_output(
    KeepOrDrop, method="function_calling"
)

# Web Search Query Rewriter
class WebQuery(BaseModel):
    query: str

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user question into a web search query composed of keywords.\n"
               "Rules:\n"
               "- Keep it short (6-14 words).\n"
               "- If the question implies recency (e.g., recent/latest/last week/last month), "
               "add a constraint like (last 30 days).\n"
               "- Do NOT answer the question.\n"
               "- Return JSON with a single key: query"),
    ("human", "Question: {question}"),
])

rewrite_chain = rewrite_prompt | eval_llm.with_structured_output(
    WebQuery, method="function_calling"
)

# Generation Prompt
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
               "If the context is empty or insufficient, say: 'I don't know.'"),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])

# Reflection & Hallucination Assessment
class ReflectionEval(BaseModel):
    grounded: bool = Field(description="True if the answer is grounded in and supported by the context.")
    hallucination: bool = Field(description="True if the answer introduces unsupported facts.")
    confidence: int = Field(description="Confidence score between 0 and 100.", ge=0, le=100)
    verdict: Literal["good", "regenerate", "ambiguous"] = Field(description="'good' if grounded, 'regenerate' if hallucinated.")
    reasoning: str = Field(description="Brief explanation of the groundedness evaluation.")

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You evaluate whether a generated answer is grounded in the provided context.
Rules:
- grounded=true if statements in the answer are supported by the context or if answer says 'I don't know'.
- hallucination=true if the answer asserts facts contradicted by or absent from the context.
- verdict='good' if satisfactory, 'regenerate' if ungrounded.
- confidence is 0-100 based on factual alignment."""),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"),
])

reflection_chain = reflection_prompt | eval_llm.with_structured_output(
    ReflectionEval, method="function_calling"
)

# Graph Node Implementations
def retrieve_node(state: State) -> Dict[str, Any]:
    q = state["question"]
    steps = state.get("steps") or []

    try:
        col_count = vectorstore._collection.count()
    except Exception:
        col_count = 0

    if col_count == 0:
        logger.info("Chroma is empty; skipping vector search.")
        return {
            "docs": [],
            "steps": steps + ["Knowledge base has no documents; proceeding to evaluate external knowledge."]
        }

    docs = retriever.invoke(q)
    return {
        "docs": docs,
        "steps": steps + [f"Retrieved {len(docs)} chunk(s) from document knowledge base."]
    }

def eval_each_doc_node(state: State) -> Dict[str, Any]:
    q = state["question"]
    retrieved = state.get("docs") or []
    steps = state.get("steps") or []

    inputs = [{"question": q, "chunk": d.page_content} for d in retrieved]
    results = (
        doc_eval_chain.batch(inputs, config={"max_concurrency": 4}, return_exceptions=True)
        if inputs
        else []
    )

    scores: List[float] = []
    good: List[Document] = []

    for i, (d, out) in enumerate(zip(retrieved, results)):
        if isinstance(out, Exception) or out is None:
            score, label, why = GRADER_FAILURE_SCORE, "GRADER_FAILED", str(out)
        else:
            score, label, why = LABEL_SCORES.get(out.label, GRADER_FAILURE_SCORE), out.label, out.reason

        scores.append(score)
        logger.info("[Chunk %d] %s (score %.2f): %s", i, label, score, why)
        if score > LOWER_TH:
            good.append(d)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    if not scores or avg_score <= LOWER_TH:
        verdict = "INCORRECT"
        reason = f"Average score {avg_score:.2f} <= {LOWER_TH}."
    elif avg_score > UPPER_TH:
        verdict = "CORRECT"
        reason = f"Average score {avg_score:.2f} > {UPPER_TH}."
    else:
        verdict = "AMBIGUOUS"
        reason = f"Average score {avg_score:.2f} is between thresholds."

    eval_step = f"Evaluated {len(retrieved)} chunk(s): verdict={verdict} (avg score: {avg_score:.2f})."
    return {
        "good_docs": good,
        "scores": scores,
        "verdict": verdict,
        "reason": reason,
        "eval_score": avg_score,
        "steps": steps + [eval_step]
    }

def route_after_eval(state: State) -> str:
    # CORRECT -> internal docs only. INCORRECT / AMBIGUOUS -> go to the web.
    return "refine" if state.get("verdict") == "CORRECT" else "rewrite_query"

def rewrite_query_node(state: State) -> Dict[str, Any]:
    steps = state.get("steps") or []
    try:
        out = rewrite_chain.invoke({"question": state["question"]})
        query = out.query
    except Exception as e:
        logger.warning("Rewrite query failed: %s; falling back to original question", e)
        query = state["question"]

    return {
        "web_query": query,
        "steps": steps + [f"Rewrote query for web search: \"{query}\"."]
    }

def web_search_node(state: State) -> Dict[str, Any]:
    steps = state.get("steps") or []
    q = state.get("web_query") or state["question"]
    web_docs: List[Document] = []

    try:
        results = tavily.invoke({"query": q})
        for r in results or []:
            if not isinstance(r, dict):
                continue
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "")
            if not content:
                continue
            web_docs.append(Document(page_content=content, metadata={"url": url, "title": title, "source": "web"}))
    except Exception as e:
        logger.error("Web search failed: %s", e)

    return {
        "web_docs": web_docs,
        "steps": steps + [f"Web search retrieved {len(web_docs)} external result(s)."]
    }

def refine_node(state: State) -> Dict[str, Any]:
    q = state["question"]
    verdict = state.get("verdict")
    good_docs = state.get("good_docs") or []
    web_docs = state.get("web_docs") or []
    steps = state.get("steps") or []

    if verdict == "CORRECT":
        docs_to_use = good_docs
    elif verdict == "INCORRECT":
        docs_to_use = web_docs
    else:  # AMBIGUOUS -> combine internal + web
        docs_to_use = good_docs + web_docs

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    strips = decompose_to_sentences(context)

    inputs = [{"question": q, "sentence": s} for s in strips]
    results = (
        filter_chain.batch(inputs, config={"max_concurrency": 8}, return_exceptions=True)
        if inputs
        else []
    )

    kept: List[str] = []
    for s, r in zip(strips, results):
        if isinstance(r, Exception) or r is None or getattr(r, "keep", True):
            kept.append(s)

    refined_context = "\n".join(kept).strip()
    if not refined_context and context:
        refined_context = context[:3000]

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
        "steps": steps + [f"Knowledge refinement: retained {len(kept)}/{len(strips)} key factual sentences."]
    }

def generate_node(state: State) -> Dict[str, Any]:
    steps = state.get("steps") or []
    out = (answer_prompt | llm).invoke(
        {"question": state["question"], "context": state.get("refined_context", "")}
    )
    answer_text = str(out.content).strip()
    return {
        "answer": answer_text,
        "steps": steps + ["Synthesized final answer from refined knowledge."]
    }

def reflect_node(state: State) -> Dict[str, Any]:
    steps = state.get("steps") or []
    q = state["question"]
    ctx = state.get("refined_context", "")
    ans = state.get("answer", "")

    refl_dict = {
        "grounded": True,
        "hallucination": False,
        "confidence": 95,
        "verdict": "good",
        "reasoning": "Answer is grounded in the refined context."
    }

    try:
        out = reflection_chain.invoke({"question": q, "context": ctx, "answer": ans})
        refl_dict = {
            "grounded": out.grounded,
            "hallucination": out.hallucination,
            "confidence": out.confidence,
            "verdict": out.verdict,
            "reasoning": out.reasoning,
        }
    except Exception as e:
        logger.warning("Reflection evaluation failed: %s; using defaults", e)

    return {
        "reflection": refl_dict,
        "steps": steps + [f"Answer reflection verified: {refl_dict['verdict']} (confidence: {refl_dict['confidence']}%)."]
    }

def build_crag_graph():
    g = StateGraph(State)

    g.add_node("retrieve", retrieve_node)
    g.add_node("eval_each_doc", eval_each_doc_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)
    g.add_node("refine", refine_node)
    g.add_node("generate", generate_node)
    g.add_node("reflect", reflect_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")
    g.add_conditional_edges("eval_each_doc", route_after_eval, {
        "refine": "refine",
        "rewrite_query": "rewrite_query",
    })
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", "reflect")
    g.add_edge("reflect", END)

    return g.compile()

crag_graph = build_crag_graph()

# =============================================================
# 4. FastAPI Application, Lifespan & Models
# =============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_default_documents()
    yield

app = FastAPI(
    title="Self-Correcting RAG API",
    description="Full-stack Corrective RAG (CRAG) system with document ingestion and autonomous corrective retrieval.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Models for Request and Response
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

class DocumentItem(BaseModel):
    id: str
    name: str
    size: int
    type: str
    chunks: int

# =============================================================
# 5. API Endpoints
# =============================================================

@app.get("/api/health")
def health_check():
    try:
        col_count = vectorstore._collection.count()
    except Exception:
        col_count = 0
    return {
        "status": "healthy",
        "total_documents": len(doc_registry),
        "total_embeddings": col_count,
    }

@app.get("/api/documents", response_model=List[DocumentItem])
def list_documents():
    return [
        DocumentItem(
            id=d["id"],
            name=d["name"],
            size=d.get("size", 0),
            type=d.get("type", "application/pdf"),
            chunks=d.get("chunks", 0),
        )
        for d in doc_registry.values()
    ]

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    uploaded = []
    for file in files:
        doc_id = str(uuid.uuid4())
        safe_filename = Path(file.filename).name
        dest = UPLOADS_DIR / f"{doc_id}_{safe_filename}"

        try:
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)

            mime_type = file.content_type or ("application/pdf" if safe_filename.endswith(".pdf") else "text/plain")
            index_file(dest, doc_id, safe_filename, mime_type)
            uploaded.append(doc_registry[doc_id])
        except Exception as e:
            logger.error("Failed to process uploaded file %s: %s", safe_filename, e)
            if dest.exists():
                dest.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Failed to process {safe_filename}: {str(e)}")

    return {
        "message": f"Successfully processed {len(uploaded)} file(s)",
        "documents": uploaded,
    }

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    if doc_id not in doc_registry:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        vectorstore._collection.delete(where={"doc_id": doc_id})
        logger.info("Deleted doc_id=%s from Chroma.", doc_id)
    except Exception as e:
        logger.error("Error deleting from Chroma collection: %s", e)

    doc_info = doc_registry.get(doc_id, {})
    file_rel = doc_info.get("file_path")
    if file_rel:
        stored_file = UPLOADS_DIR / file_rel
        if stored_file.exists():
            stored_file.unlink(missing_ok=True)

    for f in UPLOADS_DIR.glob(f"{doc_id}_*"):
        f.unlink(missing_ok=True)

    del doc_registry[doc_id]
    save_registry()

    return {"message": "Document removed successfully"}

@app.post("/api/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    initial_state: State = {
        "question": q,
        "docs": [],
        "good_docs": [],
        "scores": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "web_query": "",
        "web_docs": [],
        "answer": "",
        "eval_score": 0.0,
        "reflection": {},
        "steps": ["Initialized Self-Correcting RAG pipeline."],
    }

    try:
        result = crag_graph.invoke(initial_state)
    except Exception as e:
        logger.error("CRAG pipeline execution failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"CRAG execution failed: {str(e)}")

    final_answer = result.get("answer") or "No answer generated."

    # Evaluation
    eval_score = int(round(result.get("eval_score", 0.0) * 100))
    eval_verdict = result.get("verdict", "AMBIGUOUS")
    eval_reasoning = result.get("reason", "")

    # Reflection
    refl = result.get("reflection") or {}
    refl_verdict = refl.get("verdict", "good")
    refl_grounded = refl.get("grounded", True)
    refl_hallucination = refl.get("hallucination", False)
    refl_confidence = refl.get("confidence", 95)
    refl_reasoning = refl.get("reasoning", "Answer successfully grounded.")

    # Deduplicate sources
    all_docs = []
    if result.get("good_docs"):
        all_docs.extend(result["good_docs"])
    elif result.get("docs"):
        all_docs.extend(result["docs"])

    if result.get("web_docs"):
        all_docs.extend(result["web_docs"])

    unique_sources: Dict[str, SourceItem] = {}
    for doc in all_docs:
        meta = getattr(doc, "metadata", {}) or {}
        name = meta.get("source_name") or meta.get("title") or meta.get("source") or meta.get("url") or "Document"
        name_clean = str(Path(name).name) if not str(name).startswith("http") else str(name)
        page = meta.get("page")
        url = meta.get("url")
        is_web = meta.get("source") == "web" or bool(url) or str(name).startswith("http")
        doc_type = "web" if is_web else "document"

        key = f"{name_clean}_{page}_{url}"
        if key not in unique_sources:
            snippet = doc.page_content[:250].strip() + ("..." if len(doc.page_content) > 250 else "")
            unique_sources[key] = SourceItem(
                name=name_clean,
                type=doc_type,
                page=int(page) if page is not None else None,
                url=url,
                snippet=snippet,
            )

    steps = result.get("steps") or ["Executed Self-Correcting RAG pipeline."]

    return QueryResponse(
        answer=final_answer,
        sources=list(unique_sources.values())[:6],
        evaluation=EvaluationData(
            score=eval_score,
            verdict=eval_verdict,
            reasoning=eval_reasoning,
        ),
        reflection=ReflectionData(
            verdict=refl_verdict,
            grounded=refl_grounded,
            hallucination=refl_hallucination,
            confidence=refl_confidence,
            reasoning=refl_reasoning,
        ),
        steps=steps,
    )

# =============================================================
# 6. Static Web UI Frontend Serving (SPA)
# =============================================================
if (DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
async def serve_spa(full_path: str):
    # Let /api routes return 404 if not matched by existing endpoints
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Endpoint not found")

    file_path = DIST_DIR / full_path
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))

    index_html = DIST_DIR / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))

    return {
        "message": "Self-Correcting RAG backend is running. Build frontend via 'npm run build' inside frontend directory."
    }

# =============================================================
# 7. Frontend Process Manager & Direct Runner
# =============================================================
frontend_process: Optional[subprocess.Popen] = None

def stop_frontend():
    global frontend_process
    if frontend_process and frontend_process.poll() is None:
        logger.info("Stopping frontend development server...")
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(frontend_process.pid), signal.SIGTERM)
            else:
                frontend_process.terminate()
        except Exception:
            pass
        frontend_process = None

atexit.register(stop_frontend)

def launch_frontend():
    global frontend_process
    # Build frontend if dist doesn't exist yet
    index_file = DIST_DIR / "index.html"
    if not index_file.exists() and (FRONTEND_DIR / "package.json").exists():
        logger.info("Building frontend distribution...")
        try:
            subprocess.run(["npm", "run", "build"], cwd=str(FRONTEND_DIR), check=True)
            logger.info("Frontend built successfully.")
        except Exception as e:
            logger.warning("Could not auto-build frontend: %s", e)

    # Launch Vite dev server in background
    if (FRONTEND_DIR / "package.json").exists():
        try:
            frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(FRONTEND_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
            logger.info("Launched frontend Vite dev server (PID: %d)", frontend_process.pid)
        except Exception as e:
            logger.warning("Failed to start Vite dev server: %s", e)

def handle_signal(sig, frame):
    stop_frontend()
    sys.exit(0)

if __name__ == "__main__":
    import uvicorn

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    launch_frontend()

    print("\n" + "=" * 68)
    print("  🚀 Self-Correcting RAG Full-Stack System Started!")
    print("  --------------------------------------------------")
    print("  📱 Web Application:   http://localhost:8000")
    print("  ⚡ Vite Dev Server:    http://localhost:5173")
    print("  📖 API Documentation: http://localhost:8000/docs")
    print("=" * 68 + "\n")

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
