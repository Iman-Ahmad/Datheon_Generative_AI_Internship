import json
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from extractor import DEFAULT_MODELS, SCHEMAS, extract, ollama_client
from loaders import read_document

BASE = Path(__file__).parent
SAMPLE_DIRS = {"invoice": BASE / "samples" / "invoices", "resume": BASE / "samples" / "resumes"}

st.set_page_config(page_title="Structured JSON Extractor", page_icon="🧾", layout="wide")


@st.cache_data(ttl=30)
def installed_models() -> list[str]:
    """Models downloaded in Ollama (empty list if Ollama isn't running)."""
    try:
        return sorted(m.model for m in ollama_client.list().models)
    except Exception:
        return []


def read_upload(uploaded) -> str:
    """read_document() expects a file path, so save the upload to a temporary file first."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    try:
        return read_document(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def render_invoice(d: dict):
    st.subheader(d["vendor"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Invoice number", d["invoice_number"])
    c2.metric("Date", d["date"])
    c3.metric("Currency", d["currency"])
    c4.metric("Total", f"{d['total']:.2f}")

    items = pd.DataFrame(d["line_items"])
    items["line_total"] = items["quantity"] * items["unit_price"]
    st.dataframe(items, hide_index=True)

    subtotal = items["line_total"].sum()
    st.caption(
        f"Check: subtotal {subtotal:.2f} - discount {d['discount']:.2f} + tax {d['tax']:.2f} "
        f"= {subtotal - d['discount'] + d['tax']:.2f}, matching the document total of {d['total']:.2f}"
    )


def render_resume(d: dict):
    st.subheader(d["full_name"])
    contact = [x for x in (d["email"], d["phone"], d["location"]) if x]
    st.write(" · ".join(contact) if contact else "No contact details found")
    st.markdown("**Skills:** " + (", ".join(d["skills"]) or "none listed"))

    st.markdown("**Experience**")
    if d["experience"]:
        st.dataframe(pd.DataFrame(d["experience"]), hide_index=True)
    else:
        st.caption("No work experience listed")

    st.markdown("**Education**")
    if d["education"]:
        st.dataframe(pd.DataFrame(d["education"]), hide_index=True)
    else:
        st.caption("No education listed")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    doc_type = st.radio("Document type", list(SCHEMAS), format_func=str.capitalize, horizontal=True)
    provider = st.selectbox("Provider", ["ollama", "openai"], help="Ollama runs locally and is free. OpenAI needs OPENAI_API_KEY.")
    if provider == "ollama":
        models = installed_models()
        if models:
            default = DEFAULT_MODELS["ollama"]
            model = st.selectbox("Model", models, index=models.index(default) if default in models else 0)
        else:
            st.warning("No Ollama models found. Is Ollama running?")
            model = st.text_input("Model", DEFAULT_MODELS["ollama"])
    else:
        model = st.text_input("Model", DEFAULT_MODELS["openai"])

# ---------------- Main page ----------------
st.title("🧾 Structured LLM JSON Extractor")
st.caption("Turns unstructured invoices and resumes into strictly validated JSON.")

source = st.radio("Input", ["Sample document", "Upload file", "Paste text"], horizontal=True)
text = ""

try:
    if source == "Sample document":
        files = sorted(p for p in SAMPLE_DIRS[doc_type].iterdir() if p.suffix.lower() in {".txt", ".pdf", ".docx"})
        choice = st.selectbox("Sample", files, format_func=lambda p: p.name)
        text = read_document(str(choice)) if choice else ""
    elif source == "Upload file":
        uploaded = st.file_uploader("Upload a .txt, .pdf or .docx file", type=["txt", "pdf", "docx"])
        text = read_upload(uploaded) if uploaded else ""
    else:
        text = st.text_area("Paste the document text", height=250)
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))

if text and source != "Paste text":
    with st.expander("Text that will be sent to the model"):
        st.text(text)

if st.button("Extract", type="primary", disabled=not text.strip()):
    retries = []
    start = time.time()
    with st.spinner(f"Asking {model}... local models can take up to a minute."):
        try:
            result = extract(
                text, doc_type, model=model, provider=provider,
                on_retry=lambda attempt, reason: retries.append((attempt, reason)),
            )
            st.session_state["output"] = {
                "type": doc_type, "data": result.model_dump(), "retries": retries,
                "seconds": time.time() - start, "model": model,
            }
        except RuntimeError as e:
            st.session_state["output"] = {"error": str(e), "retries": retries}

# ---------------- Results (kept in session_state so they survive button clicks) ----------------
out = st.session_state.get("output")
if out:
    st.divider()
    if "error" in out:
        st.error(out["error"])
    else:
        st.success(f"Validated in {out['seconds']:.0f}s with {out['model']}")
        (render_invoice if out["type"] == "invoice" else render_resume)(out["data"])
        st.download_button(
            "Download JSON",
            json.dumps(out["data"], indent=2, ensure_ascii=False),
            file_name=f"{out['type']}_result.json",
            mime="application/json",
        )
        with st.expander("Raw JSON"):
            st.json(out["data"])

    if out["retries"]:
        with st.expander(f"Validation caught {len(out['retries'])} problem(s) and asked the model to fix them"):
            for attempt, reason in out["retries"]:
                st.write(f"**Attempt {attempt}:** {reason[:300]}")
    elif "error" not in out:
        st.caption("Valid on the first attempt, no retries needed.")