import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(
    page_title="Uganda Legal AI",
    page_icon="⚖️",
    layout="wide"
)

# ------------------------
# CUSTOM CSS (Modern UI)
# ------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stChatMessage {
    padding: 1rem;
    border-radius: 18px;
    margin-bottom: 10px;
}

[data-testid="stChatMessage-user"] {
    background-color: #1e293b;
    color: white;
}

[data-testid="stChatMessage-assistant"] {
    background-color: #f1f5f9;
    color: #111827;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Uganda Legal Awareness AI")
st.caption("Powered by RAG 3.0 + Groq")

# ------------------------
# LOAD MODELS & INDEX (CACHED)
# ------------------------
@st.cache_resource
def load_system():
    index = faiss.read_index("uganda_legal_index.faiss")
    
    with open("uganda_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
        
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    return index, chunks, embedder, reranker

index, all_chunks, embedding_model, reranker = load_system()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ------------------------
# RETRIEVAL
# ------------------------
def retrieve(query, top_k=8):
    query_embedding = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [all_chunks[i] for i in indices[0]]

def rerank(query, retrieved, top_k=5):
    pairs = [[query, chunk["content"]] for chunk in retrieved]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(retrieved, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:top_k]]

# ------------------------
# GENERATE ANSWER
# ------------------------
def generate_answer(query, context_chunks):
    context_text = "\n\n".join(
        [f"Source: {c['source']}\n{c['content']}" for c in context_chunks]
    )
    
    prompt = f"""
You are a Ugandan legal awareness assistant.
Provide a clear but detailed answer.
Use only the provided context.
If missing, say the information is not found.

Context:
{context_text}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1200
    )
    
    return response.choices[0].message.content

# ------------------------
# CHAT STATE
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a legal question..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Acts and generating response..."):
            
            retrieved = retrieve(prompt)
            reranked = rerank(prompt, retrieved)
            answer = generate_answer(prompt, reranked)
            
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
