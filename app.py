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
# MODERN UGANDAN CIVIC UI
# -----------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    background-color: #F4F6F9;
}

/* Remove top padding from Streamlit container */
.block-container {
    padding-top: 0rem;
}

/* Full-width header */
.full-width-header {
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    background: linear-gradient(90deg, #0B0F19 0%, #1A1F2E 100%);
    padding: 2rem 2rem;
    border-bottom: 4px solid #FFCD00;
}

.header-title {
    color: white;
    font-size: 2rem;
    font-weight: 700;
}

.header-sub {
    color: #FFCD00;
    font-size: 0.95rem;
    margin-top: 0.4rem;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------
st.markdown("""
<div class="full-width-header">
    <div class="header-title">⚖️ UGANDA LEGAL AWARENESS AI</div>
    <div class="header-sub">
        Grounded Legal Intelligence • RAG 3.0 • Powered by Groq
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
Provide a structured, clear, and detailed answer.
Use ONLY the provided context.
If not found, clearly state that the answer is not available in the provided Acts.

Use:
- Headings where helpful
- Bullet points for clarity
- Section references when available

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

    # AUTO SCROLL SCRIPT
    st.markdown("""
        <script>
            const streamlitDoc = window.parent.document;
            const main = streamlitDoc.querySelector('section.main');
            main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });
        </script>
    """, unsafe_allow_html=True)

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("""
<div class="footer">
    UgandaLegalAI • Legal Access Platform • Brian Miami
</div>
""", unsafe_allow_html=True)
