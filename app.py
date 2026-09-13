"""
DrillSafe-AI: HSE & Drilling Safety Assistant
-----------------------------------------------
A Streamlit chatbot that answers ONLY from the content of HSE_QA.TXT.txt,
using the Gemini API. Built for HSE Officers, Field Engineers, and Workers
to get instant, accurate safety guidance on-site.
"""

import os
import re
import streamlit as st
from google import genai
from google.genai import types

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
APP_TITLE = "DrillSafe-AI"
APP_TAGLINE = "HSE & Drilling Safety Assistant"
DATA_FILE = "HSE_QA.TXT.txt"
MODEL_NAME = "gemini-2.5-flash"  # fast + cheap, good for Q&A retrieval tasks
TOP_K_CHUNKS = 4                # how many Q&A entries to feed the model per question

st.set_page_config(page_title=APP_TITLE, page_icon="🦺", layout="centered")

# --------------------------------------------------------------------------
# LOAD & CHUNK THE KNOWLEDGE FILE
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_qa_chunks(path: str):
    """
    Parses HSE_QA.TXT.txt into a list of Q/A chunks.
    Expected format (repeatable):
        Q: <question>
        A: <answer>
    Falls back to paragraph-splitting if the Q:/A: pattern isn't found,
    so the app still works with plain HSE manual text.
    """
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Try Q:/A: structured parsing first
    pattern = re.compile(r"Q:\s*(.*?)\s*A:\s*(.*?)(?=\nQ:|\Z)", re.DOTALL)
    matches = pattern.findall(text)

    chunks = []
    if matches:
        for q, a in matches:
            q, a = q.strip(), a.strip()
            chunks.append({"question": q, "answer": a, "text": f"Q: {q}\nA: {a}"})
    else:
        # Fallback: split on blank lines/paragraphs
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if para:
                chunks.append({"question": "", "answer": para, "text": para})

    return chunks


def score_chunk(query: str, chunk_text: str) -> int:
    """Simple keyword-overlap scoring for retrieval (no external embedding API needed)."""
    query_words = set(re.findall(r"[a-z0-9']+", query.lower()))
    chunk_words = set(re.findall(r"[a-z0-9']+", chunk_text.lower()))
    stopwords = {"what", "is", "the", "a", "an", "to", "of", "in", "on", "for",
                 "and", "or", "do", "does", "how", "should", "i", "if", "are", "be"}
    query_words -= stopwords
    if not query_words:
        return 0
    return len(query_words & chunk_words)


def retrieve_relevant_chunks(query: str, chunks: list, top_k: int = TOP_K_CHUNKS):
    scored = [(score_chunk(query, c["text"]), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    # Only keep chunks with at least some overlap; if none score >0, still pass top few
    relevant = [c for s, c in scored if s > 0][:top_k]
    if not relevant:
        relevant = [c for _, c in scored[:top_k]]
    return relevant


# --------------------------------------------------------------------------
# GEMINI SETUP
# --------------------------------------------------------------------------
def get_api_key():
    # Priority: Streamlit secrets -> environment variable -> sidebar input
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass  # No secrets.toml file exists yet - that's fine, just skip this step
    return os.environ.get("GEMINI_API_KEY", "")


SYSTEM_INSTRUCTION = """You are DrillSafe-AI, a strict HSE (Health, Safety, Environment) assistant
for a drilling company. You must answer ONLY using the information provided in the CONTEXT below,
which comes from the company's approved HSE_QA.TXT.txt file.

Rules:
1. If the answer is fully or partially contained in the CONTEXT, answer clearly and step-by-step
   where the source uses numbered steps.
2. If the CONTEXT does not contain enough information to answer, say exactly:
   "I don't have this information in the HSE file. Please check the official manual or contact your HSE officer."
   Do NOT guess or use outside knowledge, even if you know the general answer.
3. Keep answers short, practical, and field-usable — this may be read on a phone at a drill site.
4. Never invent procedures, numbers, or thresholds that are not in the CONTEXT.
"""


def ask_gemini(client, question: str, context_chunks: list):
    context_text = "\n\n".join(c["text"] for c in context_chunks)
    prompt = f"""CONTEXT (from HSE_QA.TXT.txt):
{context_text}

QUESTION: {question}

Answer using only the CONTEXT above, following your instructions."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )
    return response.text


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title(f"🦺 {APP_TITLE}")
st.caption(APP_TAGLINE)

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=get_api_key(),
        help="Get a key from Google AI Studio. You can also set GEMINI_API_KEY as an env var or in Streamlit secrets.",
    )
    st.markdown("---")
    st.markdown("**Knowledge source:** `HSE_QA.TXT.txt`")
    if st.button("🔄 Reload knowledge file"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.caption("This assistant only answers from the company's approved HSE file. "
               "It will not use general knowledge to answer safety questions.")

chunks = load_qa_chunks(DATA_FILE)

if not chunks:
    st.error(f"Could not find or parse `{DATA_FILE}`. Please upload/place it in the app folder.")
    st.stop()

st.success(f"Loaded {len(chunks)} HSE entries from `{DATA_FILE}`.")

if "history" not in st.session_state:
    st.session_state.history = []

# Render chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Example quick-ask buttons
st.markdown("**Quick questions:**")
cols = st.columns(3)
example_qs = ["What PPE is required?", "Gas leak — what do I do?", "Steps for a JSA?"]
quick_click = None
for col, eq in zip(cols, example_qs):
    if col.button(eq):
        quick_click = eq

user_input = st.chat_input("Ask an HSE question (e.g. 'What do I do if I detect a gas leak?')")
question = quick_click or user_input

if question:
    if not api_key_input:
        st.warning("Please enter your Gemini API key in the sidebar to get an answer.")
    else:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Checking HSE file..."):
                try:
                    client = genai.Client(api_key=api_key_input)
                    relevant_chunks = retrieve_relevant_chunks(question, chunks)
                    answer = ask_gemini(client, question, relevant_chunks)
                except Exception as e:
                    answer = f"⚠️ Error contacting Gemini API: {e}"

                st.markdown(answer)

                with st.expander("📄 Source entries used"):
                    for c in relevant_chunks:
                        st.markdown(f"- {c['text'][:200]}{'...' if len(c['text']) > 200 else ''}")

        st.session_state.history.append({"role": "assistant", "content": answer})
