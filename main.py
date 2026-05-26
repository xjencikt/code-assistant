import os
import shutil
import time
import stat
from git import Repo
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load variables from .env file
load_dotenv()

# Configuration constants
INDEX_NAME = "codebase-rag"
EMBEDDING_DIMENSION = 768  # Gemini text-embedding-004 output size
SUPPORTED_EXTENSIONS = {
    ".py": "python", ".js": "js", ".ts": "ts",
    ".tsx": "ts", ".jsx": "js", ".go": "go",
    ".java": "java", ".cpp": "cpp"
}

# 1. Initialize API Clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=768
)

def init_pinecone_index():
    """Checks if index exists, if not creates a serverless index matching Gemini specs."""
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Free tier default region for Pinecone serverless
            )
        )
        # Wait a moment for cloud infrastructure provisioning to settle
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
        print("Pinecone index is ready!")
    else:
        print(f"Pinecone index '{INDEX_NAME}' already exists.")

def on_rm_error(func, path, exc_info):
    """Clears the read-only bit on Windows so shutil.rmtree can delete .git files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_repository(repo_url: str, local_dir: str) -> str:
    """Clones a GitHub repository to a local directory, handling Windows permissions safely."""
    if os.path.exists(local_dir):
        # Use our custom error handler to bypass Windows read-only locks
        shutil.rmtree(local_dir, onerror=on_rm_error)
    print(f"Cloning {repo_url}...")
    Repo.clone_from(repo_url, local_dir)
    return local_dir

def load_and_chunk_codebase(repo_dir: str):
    """Walks through the repo, reads code files, and chunks them intelligently."""
    documents = []
    
    # Standardize path separators for safety
    normalized_repo_dir = os.path.abspath(repo_dir)

    for root, dirs, files in os.walk(normalized_repo_dir):
        # Fix: ONLY skip hidden system folders like .git, don't break on relative path prefixes
        parts = os.path.relpath(root, normalized_repo_dir).split(os.sep)
        if any(part.startswith('.') and part != '.' for part in parts):
            continue
            
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, normalized_repo_dir)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code_content = f.read()
                    
                    doc = Document(
                        page_content=code_content,
                        metadata={
                            "source": relative_path,
                            "language": SUPPORTED_EXTENSIONS[file_ext]
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"Skipping file {relative_path}: {e}")

    print(f"Successfully read {len(documents)} code files from disk.")
    
    chunked_docs = []
    for doc in documents:
        lang = doc.metadata["language"]
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=1200, chunk_overlap=200
        )
        chunks = splitter.split_documents([doc])
        chunked_docs.extend(chunks)
        
    return chunked_docs

def index_codebase(repo_url: str):
    """Orchestrates cloning, chunking, and embedding with conservative rate-limit recovery."""
    local_path = "./temp_repo"
    
    # Ensure index exists
    init_pinecone_index()
    index = pc.Index(INDEX_NAME)
    
    # Extract code elements
    clone_repository(repo_url, local_path)
    chunks = load_and_chunk_codebase(local_path)
    
    print(f"Generating embeddings and uploading {len(chunks)} vectors to Pinecone...")
    
    # Drop batch size to 20 to strictly respect Tokens-Per-Minute limits
    BATCH_SIZE = 20 
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        texts_to_embed = [chunk.page_content for chunk in batch_chunks]
        
        try:
            batch_vectors = embeddings.embed_documents(texts_to_embed)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                # A 65-second sleep completely flushes out Google's sliding minute window
                print("\n[Quota Alert] Hit rate/token capacity. Pausing for 65 seconds for full reset...")
                time.sleep(65)
                print("Resuming index execution...")
                batch_vectors = embeddings.embed_documents(texts_to_embed)
            else:
                raise e
        
        vectors_to_upsert = []
        for j, vector in enumerate(batch_vectors):
            global_idx = i + j
            chunk = batch_chunks[j]
            
            vectors_to_upsert.append((
                f"chunk_{global_idx}", 
                vector, 
                {
                    "text": chunk.page_content,
                    "source": chunk.metadata["source"],
                    "language": chunk.metadata["language"]
                }
            ))
            
        index.upsert(vectors=vectors_to_upsert)
        print(f"Indexed chunks {i} to {min(i + BATCH_SIZE, len(chunks))}...")
        
        # Consistent pacing pause between batches
        time.sleep(3)
            
    print("Codebase successfully indexed in Pinecone!")
    
    if os.path.exists(local_path):
        shutil.rmtree(local_path, onerror=on_rm_error)

def query_codebase(user_query: str) -> str:
    """Finds top relevant code snippets and uses Gemini to answer questions."""
    index = pc.Index(INDEX_NAME)
    
    # 1. Convert user query to vector space using our existing embedding engine
    query_vector = embeddings.embed_query(user_query)
    
    # 2. Retrieve top 4 closest code chunks from Pinecone
    search_results = index.query(
        vector=query_vector, 
        top_k=4, 
        include_metadata=True
    )
    
    # Construct a string containing our codebase context snippets
    context_blocks = []
    for match in search_results.get("matches", []):
        meta = match.get("metadata", {})
        context_blocks.append(
            f"--- FILE: {meta.get('source')} ({meta.get('language')}) ---\n"
            f"{meta.get('text')}\n"
        )
    
    context = "\n".join(context_blocks)
    
    # 3. Setup Gemini Chat Model to compile the answer
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Using the updated, stable model string
    temperature=0.2, 
    google_api_key=os.getenv("GEMINI_API_KEY")
  )
    
    system_prompt = (
        "You are an expert software engineering assistant specializing in codebase analysis.\n"
        "Answer the user's questions utilizing ONLY the provided source code blocks as context.\n"
        "When explaining, reference exact file paths and code snippets where applicable.\n"
    )
    
    user_prompt = f"Context:\n{context}\n\nQuestion: {user_query}"
    
    # Get response from model
    response = llm.invoke([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    
    return response.content

if __name__ == "__main__":
    # 1. Comment out indexing since it's already done
    # TEST_REPO = "https://github.com/encode/uvicorn" 
    # index_codebase(TEST_REPO)
    
    # 2. Run a live RAG test query!
    print("\n--- RUNNING BOT QUERY TEST ---")
    
    sample_question = "Where is the server startup loop or connection handling located in this codebase?"
    answer = query_codebase(sample_question)
    
    print(f"\nQuestion: {sample_question}")
    print(f"\nAnswer:\n{answer}")