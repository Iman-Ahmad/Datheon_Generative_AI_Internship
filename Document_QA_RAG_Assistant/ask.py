"""Command-line interface for the RAG pipeline.

  python ask.py index                      build the vector index from the PDFs in docs/
  python ask.py ask "How many leave days?" ask one question
  python ask.py chat                       interactive session (type 'exit' to quit)
"""
import argparse
import sys

import rag


def answer_question(question: str, store, llm, show_chunks: bool = False):
    hits = rag.retrieve(store, question)
    if show_chunks:
        print("\n--- Retrieved chunks ---")
        for doc, dist in hits:
            print(f"[{doc.metadata['source']}, page {doc.metadata['page']}] distance={dist:.3f}")
            print(doc.page_content.replace("\n", " ")[:300] + "...\n")
        print("--- Answer ---")

    full = ""
    for piece in rag.stream_answer(llm, question, hits):
        print(piece, end="", flush=True)
        full += piece
    print()

    # Citations only make sense when the model actually answered
    if hits and not rag.is_refusal(full):
        print("\nSources:")
        for label in rag.format_sources(rag.cite(store, full, hits, question=question)):
            print(f"  - {label}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # avoid Windows console encoding problems

    parser = argparse.ArgumentParser(description="Document Q&A with RAG (local, Ollama + ChromaDB)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build the vector index from PDFs")
    p_index.add_argument("--docs", default=rag.DOCS_DIR, help="Folder with PDF files")

    for name in ("ask", "chat"):
        p = sub.add_parser(name, help="Ask a question" if name == "ask" else "Interactive chat")
        p.add_argument("--model", default=rag.CHAT_MODEL, help="Ollama chat model")
        p.add_argument("--show-chunks", action="store_true", help="Print the retrieved chunks")
        if name == "ask":
            p.add_argument("question")
    args = parser.parse_args()

    try:
        if args.command == "index":
            print(f"Indexing PDFs from '{args.docs}' ...")
            pages, chunks = rag.build_index(args.docs)
            print(f"Done: {pages} pages -> {chunks} chunks stored in '{rag.PERSIST_DIR}'.")
            return

        if rag.index_size() == 0:
            print("No index found. Run first:  python ask.py index")
            sys.exit(1)

        store = rag.open_store()
        llm = rag.get_llm(args.model)

        if args.command == "ask":
            answer_question(args.question, store, llm, args.show_chunks)
        else:
            print("Ask questions about your documents. Type 'exit' to quit.")
            while True:
                question = input("\nQuestion: ").strip()
                if question.lower() in {"exit", "quit", ""}:
                    break
                answer_question(question, store, llm, args.show_chunks)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}\nIs Ollama running, and are the models downloaded?\n"
              f"  ollama pull {rag.EMBED_MODEL}\n  ollama pull {rag.CHAT_MODEL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
