import os
import json
import shutil
import time
import stat
import tempfile
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from git import Repo
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
INDEX_NAME = "codebase-rag"
EMBEDDING_DIMENSION = 768
STATE_FILE = os.path.join(tempfile.gettempdir(), "codebase_rag_status.json")
SUPPORTED_EXTENSIONS = {
    ".py": "python", ".js": "js", ".ts": "ts",
    ".tsx": "ts", ".jsx": "js", ".go": "go",
    ".java": "java", ".cpp": "cpp"
}

# ── API Clients ───────────────────────────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=768
)

# ── Conversation history ──────────────────────────────────────────────────────
conversation_history: list[tuple[str, str]] = []

# ── File-based status state ───────────────────────────────────────────────────
_state_lock = threading.Lock()

DEFAULT_STATUS = {
    "is_indexing": False,
    "progress": 0,
    "total": 0,
    "repo_url": None,
    "error": None,
    "done": False,
}


def read_status() -> dict:
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_STATUS.copy()


def write_status(**kwargs):
    with _state_lock:
        current = read_status()
        current.update(kwargs)
        with open(STATE_FILE, "w") as f:
            json.dump(current, f)


# ── Lifespan (startup) ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pinecone_index()
    # Reset any stale "is_indexing=True" left from a previous crashed run
    if read_status()["is_indexing"]:
        write_status(is_indexing=False, error="Server restarted mid-indexing.")
    yield


app = FastAPI(
    title="Codebase RAG API",
    description=(
        "Query any GitHub repository using Retrieval-Augmented Generation.\n\n"
        "**Workflow:**\n"
        "1. `POST /index` — Clone & embed a GitHub repo into Pinecone\n"
        "2. `GET  /status` — Poll indexing progress\n"
        "3. `POST /query` — Ask questions about the codebase\n"
        "4. `POST /query/stream` — Same, but tokens stream in real-time\n"
        "5. `DELETE /index` — Wipe all vectors and reset"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class IndexRequest(BaseModel):
    repo_url: str

    model_config = {
        "json_schema_extra": {
            "example": {"repo_url": "https://github.com/encode/uvicorn"}
        }
    }


class QueryRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "example": {"question": "Where is the server startup loop located?"}
        }
    }


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class StatusResponse(BaseModel):
    is_indexing: bool
    progress: int
    total: int
    repo_url: str | None
    error: str | None
    done: bool


class IndexStatsResponse(BaseModel):
    total_vectors: int
    dimension: int
    metric: str


class MessageResponse(BaseModel):
    message: str
    repo_url: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def init_pinecone_index():
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)


def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clone_repository(repo_url: str, local_dir: str):
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir, onerror=on_rm_error)
    Repo.clone_from(repo_url, local_dir)


def load_and_chunk_codebase(repo_dir: str) -> list[Document]:
    documents = []
    normalized = os.path.abspath(repo_dir)

    for root, _, files in os.walk(normalized):
        parts = os.path.relpath(root, normalized).split(os.sep)
        if any(p.startswith(".") and p != "." for p in parts):
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, normalized)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": rel_path, "language": SUPPORTED_EXTENSIONS[ext]}
                ))
            except Exception:
                pass

    chunked = []
    for doc in documents:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=doc.metadata["language"], chunk_size=1200, chunk_overlap=200
        )
        chunked.extend(splitter.split_documents([doc]))
    return chunked


def run_indexing(repo_url: str):
    global conversation_history
    conversation_history.clear()
    local_path = tempfile.mkdtemp(prefix="codebase_rag_")

    try:
        write_status(is_indexing=True, done=False, error=None,
                     repo_url=repo_url, progress=0, total=0)

        # Wipe old vectors
        print("[Indexing] Clearing old vectors...")
        try:
            pc.Index(INDEX_NAME).delete(delete_all=True)
        except Exception as e:
            if "404" in str(e) or "Namespace not found" in str(e):
                print("[Indexing] Index already empty, skipping clear.")
            else:
                raise

        # Clone
        print(f"[Indexing] Cloning {repo_url}...")
        clone_repository(repo_url, local_path)
        print("[Indexing] Cloning done. Reading and chunking files...")

        # Chunk
        chunks = load_and_chunk_codebase(local_path)
        if not chunks:
            raise ValueError("No supported source files found in this repository.")
        write_status(total=len(chunks))
        print(f"[Indexing] {len(chunks)} chunks ready. Starting embedding uploads...\n")

        # Unique IDs per repo to avoid collisions
        import hashlib
        repo_slug = hashlib.md5(repo_url.encode()).hexdigest()[:8]

        index = pc.Index(INDEX_NAME)
        BATCH_SIZE = 20

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            texts = [c.page_content for c in batch]

            # Embed with rate-limit retry
            try:
                vectors = embeddings.embed_documents(texts)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print("[Indexing] Rate limit hit — pausing 65s...")
                    time.sleep(65)
                    print("[Indexing] Resuming...")
                    vectors = embeddings.embed_documents(texts)
                else:
                    raise

            to_upsert = [
                (f"{repo_slug}_chunk_{i + j}", vec, {
                    "text": batch[j].page_content,
                    "source": batch[j].metadata["source"],
                    "language": batch[j].metadata["language"],
                })
                for j, vec in enumerate(vectors)
            ]
            index.upsert(vectors=to_upsert)

            progress = min(i + BATCH_SIZE, len(chunks))
            write_status(progress=progress)
            pct = int(progress / len(chunks) * 100)
            print(f"[Indexing] {progress}/{len(chunks)} chunks ({pct}%) uploaded...")
            time.sleep(3)

        write_status(is_indexing=False, done=True)
        print(f"\n[Indexing] Done! {len(chunks)} vectors indexed.\n")

    except Exception as e:
        print(f"\n[Indexing] Error: {e}\n")
        write_status(is_indexing=False, error=str(e), done=False)

    finally:
        if os.path.exists(local_path):
            shutil.rmtree(local_path, onerror=on_rm_error)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return """<html><body style="font-family:sans-serif;padding:2rem">
    <h2>Codebase RAG API</h2>
    <p>Visit <a href="/docs">/docs</a> for the Swagger UI.</p>
    </body></html>"""


@app.post(
    "/index",
    summary="Index a GitHub repository",
    response_model=MessageResponse,
    tags=["Indexing"],
)
def index_repo(req: IndexRequest, background_tasks: BackgroundTasks):
    if read_status()["is_indexing"]:
        raise HTTPException(status_code=409, detail="Indexing already in progress.")
    background_tasks.add_task(run_indexing, req.repo_url)
    return {"message": "Indexing started.", "repo_url": req.repo_url}


@app.get(
    "/status",
    summary="Get indexing progress",
    response_model=StatusResponse,
    tags=["Indexing"],
)
def get_status():
    return read_status()


@app.get(
    "/stats",
    summary="Pinecone index statistics",
    response_model=IndexStatsResponse,
    tags=["Indexing"],
)
def get_stats():
    stats = pc.Index(INDEX_NAME).describe_index_stats()
    return {
        "total_vectors": stats.total_vector_count,
        "dimension": stats.dimension,
        "metric": "cosine",
    }


@app.delete(
    "/index",
    summary="Wipe all vectors",
    response_model=MessageResponse,
    tags=["Indexing"],
)
def delete_index():
    pc.Index(INDEX_NAME).delete(delete_all=True)
    conversation_history.clear()
    write_status(is_indexing=False, progress=0, total=0,
                 repo_url=None, error=None, done=False)
    return {"message": "Index wiped and conversation reset."}


@app.post(
    "/query",
    summary="Ask a question about the codebase",
    response_model=QueryResponse,
    tags=["Query"],
)
def query(req: QueryRequest):
    index = pc.Index(INDEX_NAME)

    # Guard: empty index
    stats = index.describe_index_stats()
    if stats.total_vector_count == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Please index a repository first.")

    query_vector = embeddings.embed_query(req.question)
    results = index.query(vector=query_vector, top_k=4, include_metadata=True)

    context_blocks, sources = [], []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        sources.append(meta.get("source", "unknown"))
        context_blocks.append(
            f"--- FILE: {meta.get('source')} ({meta.get('language')}) ---\n{meta.get('text')}\n"
        )

    context = "\n".join(context_blocks)
    system_prompt = (
        "You are an expert software engineering assistant specializing in codebase analysis.\n"
        "Answer the user's questions utilizing ONLY the provided source code blocks as context.\n"
        "When explaining, reference exact file paths and code snippets where applicable.\n"
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {req.question}"

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    history_msgs = [(role, msg) for role, msg in conversation_history]

    # Retry once on rate limit
    for attempt in range(2):
        try:
            response = llm.invoke([("system", system_prompt)] + [("user", user_prompt)])
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt == 0:
                    wait = 60
                    print(f"[Query] Rate limit hit — waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise HTTPException(
                        status_code=429,
                        detail="Gemini API quota exceeded. Please wait a minute and try again, or enable billing at https://aistudio.google.com"
                    )
            else:
                raise

    conversation_history.append(("user", req.question))
    conversation_history.append(("assistant", response.content))

    return {"answer": response.content, "sources": list(dict.fromkeys(sources))}

@app.post(
    "/query/stream",
    summary="Ask a question (streaming)",
    tags=["Query"],
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
def query_stream(req: QueryRequest):
    index = pc.Index(INDEX_NAME)
    query_vector = embeddings.embed_query(req.question)
    results = index.query(vector=query_vector, top_k=4, include_metadata=True)

    context_blocks, sources = [], []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        sources.append(meta.get("source", "unknown"))
        context_blocks.append(
            f"--- FILE: {meta.get('source')} ({meta.get('language')}) ---\n{meta.get('text')}\n"
        )

    context = "\n".join(context_blocks)
    system_prompt = (
        "You are an expert software engineering assistant specializing in codebase analysis.\n"
        "Answer ONLY from the provided source code context. Reference file paths when relevant.\n"
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {req.question}"

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    history_msgs = [(role, msg) for role, msg in conversation_history]

    def token_generator():
        full_response = ""
        for chunk in llm.stream([("system", system_prompt)] + [("user", user_prompt)]):
            token = chunk.content
            full_response += token
            yield f"data: {token}\n\n"

        conversation_history.append(("user", req.question))
        conversation_history.append(("assistant", full_response))
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.delete(
    "/conversation",
    summary="Clear conversation history",
    response_model=MessageResponse,
    tags=["Query"],
)
def clear_conversation():
    conversation_history.clear()
    return {"message": "Conversation history cleared."}