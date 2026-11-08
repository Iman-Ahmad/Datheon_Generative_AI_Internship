"""Core RAG pipeline.

PDF pages -> chunks -> embeddings -> ChromaDB -> retrieval -> answer from the retrieved context only,
with source page citations. Everything runs locally with Ollama.
"""
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- Settings (change here, or override from the command line / app) ----------------
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:7b"          # accuracy matters more than speed for strict Q&A; llama3.2:3b is faster
DOCS_DIR = "docs"
PERSIST_DIR = "chroma_db"
COLLECTION = "documents"

CHUNK_SIZE = 600                   # characters per chunk
CHUNK_OVERLAP = 100                # characters shared between neighbouring chunks
TOP_K = 4                          # how many chunks to retrieve per question
MAX_DISTANCE = 0.6                 # chunks farther than this are ignored (cosine distance, 0 = identical).
                                   # Tune it with evaluate.py: see the "best distance" column.
CITE_MARGIN = 0.03                 # when citing, keep chunks about as close to the ANSWER as the best one

REFUSAL = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You answer questions strictly from the context provided by the user.

Rules:
- Use ONLY the context. Never use outside knowledge, even for facts you know.
- If the context does not contain the answer, reply exactly: {REFUSAL}
- Be concise and state exact figures, names and dates as written in the context.
- Do not mention "the context" or source numbers in your answer; citations are added separately."""


# ---------------- Embeddings and models ----------------
class PrefixedOllamaEmbeddings(OllamaEmbeddings):
    """nomic-embed-text works best when documents and queries carry a task prefix."""

    def embed_documents(self, texts):
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text):
        return super().embed_query(f"search_query: {text}")


def get_embeddings(model: str = EMBED_MODEL):
    cls = PrefixedOllamaEmbeddings if model.startswith("nomic") else OllamaEmbeddings
    return cls(model=model)


def get_llm(model: str = CHAT_MODEL, temperature: float = 0.0, num_ctx: int = 6144) -> ChatOllama:
    return ChatOllama(model=model, temperature=temperature, num_ctx=num_ctx)


# ---------------- Indexing ----------------
def load_pdfs(docs_dir: str = DOCS_DIR) -> list[Document]:
    """Load every PDF in a folder, one Document per page. Page numbers are 1-based (like a PDF viewer)."""
    pages: list[Document] = []
    pdf_files = sorted(Path(docs_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{docs_dir}'.")
    for pdf in pdf_files:
        loaded = PyPDFLoader(str(pdf)).load()
        if not any(p.page_content.strip() for p in loaded):
            print(f"Warning: '{pdf.name}' has no extractable text (scanned PDF?). Skipped.")
            continue
        for p in loaded:
            p.metadata = {"source": pdf.name, "page": int(p.metadata.get("page", 0)) + 1}
        pages.extend(loaded)
    return pages


def split_documents(pages: list[Document]) -> list[Document]:
    """Cut pages into overlapping chunks. Each chunk keeps its source file and page number."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    return [c for c in chunks if c.page_content.strip()]


def open_store(persist_dir: str = PERSIST_DIR, embed_model: str = EMBED_MODEL) -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(embed_model),
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},   # scores become cosine distances (0 = identical)
    )


def build_index(docs_dir: str = DOCS_DIR, persist_dir: str = PERSIST_DIR,
                embed_model: str = EMBED_MODEL) -> tuple[int, int]:
    """(Re)build the vector store from all PDFs in docs_dir. Returns (number of pages, number of chunks)."""
    pages = load_pdfs(docs_dir)
    chunks = split_documents(pages)

    store = open_store(persist_dir, embed_model)
    try:
        store.delete_collection()          # start from a clean index so old documents don't linger
    except Exception:
        pass
    store = open_store(persist_dir, embed_model)
    store.add_documents(chunks)
    return len(pages), len(chunks)


def index_size(persist_dir: str = PERSIST_DIR, embed_model: str = EMBED_MODEL) -> int:
    """Number of chunks currently stored (0 if there is no index yet)."""
    if not Path(persist_dir).exists():
        return 0
    try:
        return len(open_store(persist_dir, embed_model).get()["ids"])
    except Exception:
        return 0


def delete_index(persist_dir: str = PERSIST_DIR):
    shutil.rmtree(persist_dir, ignore_errors=True)


# ---------------- Retrieval ----------------
def retrieve(store: Chroma, question: str, k: int = TOP_K,
             max_distance: float = MAX_DISTANCE) -> list[tuple[Document, float]]:
    """Return the k most similar chunks as (chunk, distance) pairs, best first."""
    results = store.similarity_search_with_score(question, k=k)
    return [(doc, dist) for doc, dist in results if dist <= max_distance]


def cite(store: Chroma, answer: str, hits: list[tuple[Document, float]], question: str = "",
         margin: float = CITE_MARGIN) -> list[tuple[Document, float]]:
    """Choose which retrieved chunks to cite: the ones that best SUPPORT THE ANSWER.

    Ranking by similarity to the question is not enough: a short question like "What does error code E2
    mean?" can rank an unrelated chunk first. So we search again using the answer text as the query,
    keep only chunks that were really given to the model, and cite those closest to the answer.
    The question is added to the query because short answers ("EUR 450.") carry too little meaning
    on their own to find the right chunk.
    """
    if not hits:
        return []
    in_context = {doc.page_content for doc, _ in hits}
    scored = [(doc, dist) for doc, dist in store.similarity_search_with_score(f"{question}\n{answer}".strip(), k=max(TOP_K * 3, 12))
              if doc.page_content in in_context]
    if not scored:                       # the answer matched nothing from the context: fall back to the best hit
        return hits[:1]
    best = min(dist for _, dist in scored)
    return [(doc, dist) for doc, dist in scored if dist <= best + margin]


def format_sources(cited: list[tuple[Document, float]]) -> list[str]:
    """Unique 'file, page N' labels in order of relevance."""
    seen, labels = set(), []
    for doc, _ in cited:
        label = f"{doc.metadata['source']}, page {doc.metadata['page']}"
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


# ---------------- Answering ----------------
def format_context(hits: list[tuple[Document, float]]) -> str:
    return "\n\n".join(
        f"[{doc.metadata['source']}, page {doc.metadata['page']}]\n{doc.page_content}" for doc, _ in hits
    )


def build_messages(question: str, hits: list[tuple[Document, float]],
                   history: list[dict] | None = None, memory_limit: int = 6) -> list:
    """System rules, recent conversation (optional), then the question together with the retrieved context."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in (history or [])[-memory_limit:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        messages.append(cls(content=m["content"]))
    messages.append(HumanMessage(content=f"Context:\n{format_context(hits)}\n\nQuestion: {question}"))
    return messages


def stream_answer(llm: ChatOllama, question: str, hits: list[tuple[Document, float]],
                  history: list[dict] | None = None, memory_limit: int = 6):
    """Yield the answer piece by piece. With no relevant chunks, refuse without calling the model."""
    if not hits:
        yield REFUSAL
        return
    for chunk in llm.stream(build_messages(question, hits, history, memory_limit)):
        yield chunk.content


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(marker in a for marker in ("don't know", "do not know", "not contain", "no information"))


def condense_question(llm: ChatOllama, history: list[dict], question: str) -> str:
    """Rewrite a follow-up like 'and abroad?' into a standalone question, so retrieval still works."""
    if not history:
        return question
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = (
        "Rewrite the final user question as a standalone question, using the conversation for context. "
        "Return only the rewritten question, nothing else.\n\n"
        f"Conversation:\n{transcript}\n\nFinal question: {question}\n\nStandalone question:"
    )
    rewritten = llm.invoke(prompt).content.strip().strip('"')
    return rewritten or question
