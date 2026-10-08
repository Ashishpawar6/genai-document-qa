"""Streamlit chat UI.   streamlit run app/chat.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `streamlit run app/chat.py` from anywhere

from docqa.config import load_settings  # noqa: E402
from docqa.pipeline import build_qa, build_store, download_demo, ingest_bytes, ingest_path  # noqa: E402

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
settings = load_settings()


@st.cache_resource(show_spinner="Loading the embedding model (first run downloads about 65 MB)...")
def get_store():
    return build_store(settings)


store = get_store()
qa = build_qa(settings, store)

with st.sidebar:
    st.header("Documents")
    if settings.has_api_key:
        st.success(f"Full AI mode\n\n{settings.anthropic_model} writes answers from your documents.")
    else:
        st.info("Search-only mode\n\nNo API key found, so the app shows the best matching passages. "
                "Add `ANTHROPIC_API_KEY` to your `.env` file for written answers (see the README).")

    uploads = st.file_uploader("Upload PDF, TXT or Markdown", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Add uploaded files", disabled=not uploads):
        for upload in uploads:
            try:
                total, new = ingest_bytes(store, settings, upload.name, upload.getvalue())
                st.write(f"✅ {upload.name}: {total} chunks ({new} new)")
            except ValueError as error:
                st.error(f"{upload.name}: {error}")
    if st.button("Load demo paper (Attention Is All You Need)"):
        with st.spinner("Downloading and indexing..."):
            total, new = ingest_path(store, settings, download_demo(settings))
        st.write(f"✅ {total} chunks ({new} new)")

    st.subheader("In the store")
    stored = store.sources()
    if stored:
        for name, chunks in stored.items():
            st.write(f"📄 {name} ({chunks} chunks)")
    else:
        st.caption("Nothing yet. Upload a file or load the demo paper.")
    if stored and st.button("Clear all documents"):
        store.reset()
        st.session_state.pop("history", None)
        st.rerun()

st.title("📄 Document Q&A")
st.caption("Ask questions about your documents. Answers cite their sources.")

history = st.session_state.setdefault("history", [])
for entry in history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["text"])
        if entry.get("sources"):
            with st.expander(f"Sources ({len(entry['sources'])})"):
                for s in entry["sources"]:
                    st.markdown(f"**[{s.number}] {s.label}** · similarity {s.score:.2f}")
                    st.caption(" ".join(s.text.split())[:500])

question = st.chat_input("Ask a question about your documents")
if question:
    history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer = qa.ask(question)
        st.markdown(answer.text)
        if answer.note:
            st.caption(answer.note)
        if answer.sources:
            with st.expander(f"Sources ({len(answer.sources)})", expanded=answer.mode == "retrieval-only"):
                for s in answer.sources:
                    st.markdown(f"**[{s.number}] {s.label}** · similarity {s.score:.2f}")
                    st.caption(" ".join(s.text.split())[:500])
    history.append({"role": "assistant", "text": answer.text, "sources": answer.sources})
