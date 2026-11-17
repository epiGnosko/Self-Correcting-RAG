import os
import re
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

# ==========================================
# 1. Setup & Configuration
# ==========================================
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

# Use a general instruction-following model for judging and generating.
# (schematron is an HTML->JSON extraction model and is a poor relevance judge.)
GEN_MODEL = os.getenv("GEN_MODEL", "openai/gpt-4o-mini")
EVAL_MODEL = os.getenv("EVAL_MODEL", "openai/gpt-4o-mini")

# CRAG thresholds. The grader returns a label that maps to a score below,
# so these thresholds sit cleanly between the label scores.
UPPER_TH = 0.7
LOWER_TH = 0.3
LABEL_SCORES = {"DIRECT": 0.9, "PARTIAL": 0.5, "IRRELEVANT": 0.0}
GRADER_FAILURE_SCORE = 0.5  # fail-safe: unknown -> hybrid (AMBIGUOUS) path

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

# Initialize data & vector store
docs = PyPDFLoader("2401.15884v3.pdf").load()

chunks = RecursiveCharacterTextSplitter(
    chunk_size=900, chunk_overlap=150
).split_documents(docs)
for d in chunks:
    d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=API_KEY,
    base_url=BASE_URL,
)
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# ==========================================
# 2. Graph State Definition
# ==========================================
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


# ==========================================
# 3. Graph Nodes & Prompts
# ==========================================
def retrieve_node(state: State) -> State:
    return {"docs": retriever.invoke(state["question"])}


# ---------- Retrieval evaluator ----------
class DocEval(BaseModel):
    # NOTE: field order matters. The model writes its reasoning FIRST,
    # then commits to a label, so the label follows from the reasoning.
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


def eval_each_doc_node(state: State) -> State:
    q = state["question"]
    retrieved = state["docs"]

    print("\n========== DOCUMENT EVALUATION ==========")

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
            score, label, why = LABEL_SCORES[out.label], out.label, out.reason

        scores.append(score)
        print(f"[Chunk {i}] {label} (score {score:.2f})")
        print(f"Reason: {why}\n")

        if score > LOWER_TH:
            good.append(d)

    print("ALL DOCUMENT SCORES:", scores)
    print("MAX SCORE:", max(scores) if scores else "No documents")

    # CRAG decision rule
    
    # CRAG decision rule: use average relevance
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

    print("AVERAGE SCORE:", round(avg_score, 2))
    print("VERDICT:", verdict)
    print("REASON:", reason)



    return {"good_docs": good, "scores": scores, "verdict": verdict, "reason": reason}


# ---------- Knowledge refinement ----------
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


def refine(state: State) -> State:
    q = state["question"]
    verdict = state.get("verdict")
    good_docs = state.get("good_docs", [])
    web_docs = state.get("web_docs", [])

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
        # If the filter call fails, keep the sentence rather than silently losing it.
        if isinstance(r, Exception) or r is None or r.keep:
            kept.append(s)

    refined_context = "\n".join(kept).strip()

    # Fallback: never hand the generator an empty context if we had material.
    if not refined_context and context:
        refined_context = context[:3000]

    return {"strips": strips, "kept_strips": kept, "refined_context": refined_context}


# ---------- Web search ----------
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


def rewrite_query_node(state: State) -> State:
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}


def web_search_node(state: State) -> State:
    q = state.get("web_query") or state["question"]
    results = tavily.invoke({"query": q})
    web_docs: List[Document] = []

    for r in results or []:
        if not isinstance(r, dict):
            continue
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        if not content:
            continue
        # Keep only the content in page_content so title/URL text doesn't pollute
        # sentence filtering; source info lives in metadata.
        web_docs.append(Document(page_content=content, metadata={"url": url, "title": title}))

    return {"web_docs": web_docs}


# ---------- Generation ----------
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
               "If the context is empty or insufficient, say: 'I don't know.'"),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])


def generate(state: State) -> State:
    out = (answer_prompt | llm).invoke(
        {"question": state["question"], "context": state["refined_context"]}
    )
    return {"answer": out.content}


# ==========================================
# 4. Graph Construction & Routing
# ==========================================
def route_after_eval(state: State) -> str:
    # CORRECT -> internal docs only. INCORRECT / AMBIGUOUS -> go to the web.
    return "refine" if state["verdict"] == "CORRECT" else "rewrite_query"


def build_graph():
    g = StateGraph(State)

    g.add_node("retrieve", retrieve_node)
    g.add_node("eval_each_doc", eval_each_doc_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)
    g.add_node("refine", refine)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")
    g.add_conditional_edges("eval_each_doc", route_after_eval, {
        "refine": "refine",
        "rewrite_query": "rewrite_query",
    })
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)

    return g.compile()


def initial_state(question: str) -> State:
    return {
        "question": question,
        "docs": [], "good_docs": [], "scores": [],
        "verdict": "", "reason": "",
        "strips": [], "kept_strips": [], "refined_context": "",
        "web_query": "", "web_docs": [], "answer": "",
    }


# ==========================================
# 5. Main Execution Block
# ==========================================

if __name__ == "__main__":
    app = build_graph()

    # Test an unrelated question
    question = "Compare Corrective RAG with Self-RAG in terms of runtime latency and token efficiency"

    print("\n" + "#" * 70)
    
    print("#" * 70)

    
    print("QUESTION:", question)

    res = app.invoke(initial_state(question))

    print("\n========== TEST RESULTS ==========")
    print("ACTUAL VERDICT:", res["verdict"])
    print("REASON:", res["reason"])
    print("DOCUMENT SCORES:", res["scores"])

    

    print("\nWEB QUERY:", res["web_query"])
    print("\nFINAL ANSWER:", res["answer"])
