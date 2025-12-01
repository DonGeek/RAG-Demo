import streamlit as st
import os
import tempfile

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


# ----------------- STREAMLIT CONFIG -----------------
st.set_page_config(page_title="RAG File Q&A", layout="wide")
st.title("📄 RAG Q&A — Upload a File & Ask Questions")


# ----------------- SESSION STATE -----------------
if "vector_built" not in st.session_state:
    st.session_state.vector_built = False

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = "faiss_store"


# ----------------- FILE UPLOADER -----------------
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "txt", "docx", "csv", "xlsx", "json"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

# ----------------- VECTOR BUILD PROCESS -----------------
if uploaded_files and not st.session_state.vector_built:
    with st.spinner("📥 Processing and embedding the document..."):

        # Save uploaded files into temp directory
        temp_dir = tempfile.mkdtemp()
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

        #  Load and embed
        docs = load_all_documents(temp_dir)
        store = FaissVectorStore(persist_dir=st.session_state.persist_dir)
        store.build_from_documents(docs)

        st.session_state.vector_built = True
        st.success("Vector store built successfully! Now ask your question.")


# ----------------- RESET BUTTON -----------------
if st.session_state.vector_built:
    if st.button("📂 Upload New Document"):
        st.session_state.vector_built = False
        st.session_state.uploader_key += 1   # Force uploader reset
        st.rerun()


# ----------------- QUERY SECTION -----------------
if st.session_state.vector_built:
    query = st.text_input("Ask a question based on the uploaded document:")

    if query:
        rag = RAGSearch(persist_dir=st.session_state.persist_dir)

        with st.spinner(" Searching and generating answer..."):
            answer = rag.search_and_summarize(query)

        st.write("###Answer:")
        st.write(answer)
