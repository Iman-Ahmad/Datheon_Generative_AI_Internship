"""Interactive AI chatbot with streaming, conversation memory and optional document Q&A (RAG).

Run with:  streamlit run app.py
"""
import time
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from ollama import Client

import rag

BASE = Path(__file__).parent
SAMPLE_DIR = BASE / "sample_docs"
UPLOAD_DIR = BASE / "uploads"
PERSIST_DIR = str(BASE / "chroma_db")

DEFAULT_SYSTEM_PROMPT = "You are a helpful, friendly assistant. Answer clearly and concisely."
PREFERRED_MODEL = "llama3.2:3b"

st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="centered")


# ---------------- Helpers ----------------
@st.cache_data(ttl=30)
def installed_models() -> list[str]:
    """Models downloaded in Ollama (empty list if Ollama isn't running)."""
    try:
        return sorted(m.model for m in Client().list().models)
    except Exception:
        return []


def build_plain_messages(history: list[dict], system_prompt: str, limit: int) -> list:
    """Normal chat: system prompt + the last `limit` messages."""
    messages = [SystemMessage(content=system_prompt)]
    for m in history[-limit:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        messages.append(cls(content=m["content"]))
    return messages


def stream_plain(llm, messages: list):
    for chunk in llm.stream(messages):
        yield chunk.content


def show_evidence(evidence: list[dict]):
    """Expander listing the retrieved passages with file, page and distance."""
    with st.expander(f"📄 Sources ({len(evidence)} passages)"):
        for e in evidence:
            st.markdown(f"**{e['label']}**  ·  distance {e['distance']:.2f}")
            st.caption(e["text"])


def index_folder(folder: Path):
    """Build the vector index from a folder of PDFs and report the result in the sidebar."""
    try:
        with st.spinner("Indexing documents (embedding every chunk)..."):
            pages, chunks = rag.build_index(str(folder), PERSIST_DIR)
        st.success(f"Indexed {pages} pages into {chunks} chunks.")
    except Exception as e:
        st.error(f"Indexing failed: {e}")


# ---------------- Session state: survives Streamlit's re-runs ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    models = [m for m in installed_models() if "embed" not in m]
    if not models:
        st.error("No Ollama chat models found. Start Ollama and run: ollama pull llama3.2:3b")
        st.stop()
    default_index = next((i for i, m in enumerate(models) if m.startswith(PREFERRED_MODEL)), 0)
    model = st.selectbox("Chat model", models, index=default_index)
    temperature = st.slider("Creativity (normal chat only)", 0.0, 1.5, 0.7, 0.1,
                            help="Document answers always use 0 so they stay factual.")
    memory_limit = st.slider("Messages remembered", 2, 30, 10, help="Older messages are not sent to the model.")
    system_prompt = st.text_area("System prompt (normal chat)", DEFAULT_SYSTEM_PROMPT, height=90)

    st.divider()
    st.subheader("Documents (RAG)")
    chunk_count = rag.index_size(PERSIST_DIR)
    if not any(m.startswith(rag.EMBED_MODEL) for m in installed_models()):
        st.warning(f"Embedding model missing. Run: ollama pull {rag.EMBED_MODEL}")

    uploaded = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Index uploaded PDFs", disabled=not uploaded, use_container_width=True):
        UPLOAD_DIR.mkdir(exist_ok=True)
        for old in UPLOAD_DIR.glob("*.pdf"):
            old.unlink()
        for f in uploaded:
            (UPLOAD_DIR / Path(f.name).name).write_bytes(f.getvalue())
        index_folder(UPLOAD_DIR)
        st.rerun()
    if SAMPLE_DIR.exists() and st.button("Index the sample documents", use_container_width=True):
        index_folder(SAMPLE_DIR)
        st.rerun()

    if chunk_count:
        st.caption(f"Index ready: {chunk_count} chunks")
    else:
        st.caption("No documents indexed yet.")
    rag_on = st.toggle("Answer from my documents", value=chunk_count > 0, disabled=chunk_count == 0)

    st.divider()
    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"{len(st.session_state.messages)} messages in this chat")

# ---------------- Main page ----------------
st.title("💬 AI Chatbot")
mode = "answers only from your documents" if rag_on else "normal chat"
st.caption(f"Runs fully on your computer with Ollama · {model} · {mode}")

if not st.session_state.messages:
    if rag_on:
        st.info("Ask about your documents, for example: 'How many paid annual leave days do employees get?' "
                "Then try a follow-up like 'And for part-time staff?' to see conversation memory at work.")
    else:
        st.info("Say hello! Try: 'My name is Sam. Remember it.' and then ask 'What is my name?' "
                "To chat with PDFs, index some in the sidebar first.")

# Redraw the whole history (the script re-runs from the top on every interaction)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("evidence"):
            show_evidence(m["evidence"])

if prompt := st.chat_input("Type your message..."):
    history = list(st.session_state.messages)          # everything before this question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    evidence = []
    with st.chat_message("assistant"):
        try:
            standalone = prompt
            if rag_on:
                llm = rag.get_llm(model, temperature=0.0)
                store = rag.open_store(PERSIST_DIR)
                with st.spinner("Searching the documents..."):
                    standalone = rag.condense_question(llm, history, prompt)   # makes follow-ups searchable
                    hits = rag.retrieve(store, standalone)
                if standalone != prompt:
                    st.caption(f"Searching for: {standalone}")
                stream = rag.stream_answer(llm, prompt, hits, history, memory_limit)
            else:
                hits = []
                llm = rag.get_llm(model, temperature=temperature)
                stream = stream_plain(llm, build_plain_messages(history + [{"role": "user", "content": prompt}],
                                                                system_prompt, memory_limit))

            start = time.time()
            reply = st.write_stream(stream)
            st.caption(f"{time.time() - start:.1f}s")

            if hits and not rag.is_refusal(reply):          # no sources for a refusal
                evidence = [
                    {"label": f"{d.metadata['source']}, page {d.metadata['page']}", "distance": s,
                     "text": d.page_content}
                    for d, s in rag.cite(store, reply, hits, question=standalone)
                ]
                show_evidence(evidence)
        except Exception as e:
            reply = None
            st.error(f"Could not get an answer. Is Ollama running? ({e})")

    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply, "evidence": evidence})
