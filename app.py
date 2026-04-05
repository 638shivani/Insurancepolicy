
import streamlit as st
import google.generativeai as genai
import PyPDF2
import json
import os
import numpy as np
import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PolicyMind RAG", layout="wide")

# ---------------- GEMINI SETUP ----------------
API_KEY = "YOUR_GEMINI_API_KEY"   # 🔴 Replace this
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------- EMBEDDING MODEL ----------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- FUNCTIONS ----------------

def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    
    return index

def retrieve_chunks(query, chunks, index, top_k=3):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)
    
    return [chunks[i] for i in indices[0]]

def analyze_claim_rag(query, chunks, index):
    relevant_chunks = retrieve_chunks(query, chunks, index)
    context = "\n".join(relevant_chunks)

    prompt = f"""
    You are an insurance expert.

    Policy Context:
    {context}

    User Query:
    {query}

    Output STRICT JSON:
    {{
        "decision": "Approved/Rejected",
        "reason": "Explain based ONLY on policy",
        "confidence": "X%",
        "procedure": "Medical procedure"
    }}
    """

    try:
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except:
        return {
            "decision": "Error",
            "reason": "AI issue",
            "confidence": "0%",
            "procedure": "N/A"
        }

# ---------------- STREAMLIT UI ----------------

st.title("🧠 PolicyMind RAG System")

# Session State
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload Policy PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text(uploaded_file)
        chunks = chunk_text(text)
        index = create_vector_store(chunks)

        st.session_state.chunks = chunks
        st.session_state.index = index

        st.success("✅ RAG System Ready!")

# Query Input
query = st.text_input("💬 Ask your insurance query:")

# Analyze Button
if st.button("🚀 Analyze"):
    if st.session_state.chunks and query:
        with st.spinner("Analyzing with RAG..."):
            result = analyze_claim_rag(
                query,
                st.session_state.chunks,
                st.session_state.index
            )

        st.subheader("📊 Result")

        if result["decision"] == "Approved":
            st.success(result["decision"])
        elif result["decision"] == "Rejected":
            st.error(result["decision"])
        else:
            st.warning(result["decision"])

        st.write("**Reason:**", result["reason"])
        st.write("**Confidence:**", result["confidence"])
        st.write("**Procedure:**", result["procedure"])

    else:
        st.warning("⚠ Upload PDF and enter query first")
