"""Minimal Streamlit UI for the live /ingest + /ask (RAG) FastAPI service.

Run:
  streamlit run rag_demo.py

Points at RAG_API_BASE_URL if set, else defaults to localhost — override
either way in the sidebar. This page only calls the API and displays what
it returns; all retrieval/generation logic lives in the FastAPI service.
"""

import os

import httpx
import streamlit as st

DEFAULT_API_BASE_URL = os.environ.get("RAG_API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("RAG Demo — /ingest + /ask")

api_base_url = st.sidebar.text_input("API base URL", DEFAULT_API_BASE_URL).rstrip("/")

tab_ingest, tab_ask = st.tabs(["📥 Ingest", "💬 Ask"])

with tab_ingest:
    st.subheader("POST /ingest")

    document_id = st.text_input("document_id", placeholder="e.g. my-notes")
    source = st.text_input("source (optional)", placeholder="e.g. notes.txt")
    text = st.text_area("Text to ingest", height=200, placeholder="Paste text here…")

    if st.button("Ingest", type="primary"):
        if not document_id.strip() or not text.strip():
            st.error("document_id and text are both required.")
        else:
            try:
                response = httpx.post(
                    f"{api_base_url}/ingest",
                    json={"document_id": document_id, "text": text, "source": source or None},
                    timeout=60.0,
                )
                if response.status_code == 200:
                    st.success(f"Ingested — {response.json()['chunks_indexed']} chunk(s) indexed.")
                else:
                    st.error(f"HTTP {response.status_code}")
                st.json(response.json())
            except httpx.HTTPError as exc:
                st.error(f"Request failed: {exc}")

with tab_ask:
    st.subheader("POST /ask")

    question = st.text_input("Question", placeholder="Ask something about the ingested docs…")

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.error("Question must not be empty.")
        else:
            try:
                response = httpx.post(
                    f"{api_base_url}/ask", json={"question": question}, timeout=60.0
                )
            except httpx.HTTPError as exc:
                st.error(f"Request failed: {exc}")
                response = None

            if response is not None:
                if response.status_code != 200:
                    st.error(f"HTTP {response.status_code}")
                    st.json(response.json())
                else:
                    data = response.json()
                    answer = data["answer"]

                    if answer["sources_needed"]:
                        st.warning("⚠️ Refused — insufficient context in the retrieved documents.")
                    else:
                        st.success("✅ Answered from retrieved context.")

                    st.write(answer["answer"])
                    st.caption(f"confidence: {answer['confidence']}")

                    st.markdown("**Retrieved chunk IDs (citations):**")
                    st.code("\n".join(data["retrieved_chunk_ids"]) or "(none)")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("tokens_used", data["tokens_used"])
                    c2.metric("cost_usd", f"${data['cost_usd']:.6f}")
                    c3.metric("latency_ms", data["latency_ms"])
                    c4.metric("model", data["model"])

                    with st.expander("Full raw JSON response"):
                        st.json(data)
