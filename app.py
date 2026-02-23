import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Uganda Legal AI",
    page_icon="⚖️",
    layout="wide",
)

# -----------------------------------
# MODERN UGANDAN THEME UI
# -----------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    background-color: #F8FAFC;
}

/* Header */
.header-container {
    background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
    padding: 1.2rem 2rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 1.5rem;
}

.header-title {
    font-size: 1.8rem;
    font-weight: 700;
}

.header-sub {
    font-size: 0.95rem;
    opacity: 0.8;
}

/* Chat container width */
.block-container {
    max-width: 900px;
    padding-top: 1rem;
}

/* Chat bubbles */
[data-testid="stChatMessage-user"] {
    background: linear-gradient(135deg, #0F172A, #1E293B);
    color: white;
    border-radius: 18px;
    padding: 1rem;
    border-left: 4px solid #FACC15;
}

[data-testid="stChatMessage-assistant"] {
    background: white;
    color: #111827;
    border-radius: 18px;
    padding: 1rem;
    border-left: 4px solid #B91C1C;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.05);
}

/* Input box styling */
textarea {
    border-radius: 12px !important;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.85rem;
    margin-top: 3rem;
    opacity: 0.6;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-title">⚖️ Uganda Legal Awareness AI</div>
    <div class="header-sub">
        AI-powered access to Ugandan law • RAG 3.0 • Grounded in Official Acts
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD SYSTEM (CACHED)
# -----------------------------------
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

# -----------------------------------
# RETRIEVAL
# -----------------------------------
def retrieve(query, top_k=8):
    query_embedding = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [all_chunks[i] for i in indices[0]]

def rerank(query, retrieved, top_k=5):
    pairs = [[query, chunk["content"]] for chunk in retrieved]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(retrieved, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:top_k]]

# -----------------------------------
# GENERATE ANSWER
# -----------------------------------
def generate_answer(query, context_chunks):
    context_text = "\n\n".join(
        [f"Source: {c['source']}\n{c['content']}" for c in context_chunks]
    )
    
    prompt = f"""
You are a Ugandan legal awareness assistant.
Provide a structured, clear and detailed answer.
Use ONLY the provided context.
If the answer is not found, state clearly that it is not available in the provided Acts.

Where possible:
- Reference relevant sections
- Use bullet points for clarity
- Avoid speculation

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

# -----------------------------------
# CHAT STATE
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------------
# USER INPUT
# -----------------------------------
if prompt := st.chat_input("Ask a question about Ugandan law..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing legislation and generating response..."):
            
            retrieved = retrieve(prompt)
            reranked = rerank(prompt, retrieved)
            answer = generate_answer(prompt, reranked)
            
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("""
<div class="footer">
    UgandaLegalAI • Built with FAISS, SentenceTransformers & Groq • Community Legal Awareness
</div>
""", unsafe_allow_html=True)
