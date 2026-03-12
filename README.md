# RAG AI Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built with LangChain, Groq, and Streamlit. The chatbot answers questions grounded in the content of a PDF document by combining semantic search over a vector store with a large language model.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Complete Project Flow](#complete-project-flow)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Phase Comparison](#phase-comparison)

---

## Overview

This project demonstrates how to build a document-aware AI chatbot in three incremental phases:

| Phase | File | Description |
|-------|------|-------------|
| Phase 2 | `phase2.py` | LLM-powered chatbot (Streamlit UI + Groq, no document context) |
| Phase 3 | `phase3.py` | Full RAG pipeline with PDF knowledge base |

The final implementation (`phase3.py`) lets you chat with a PDF document. It splits the document into chunks, embeds them into a vector store, and retrieves the most relevant chunks at query time before sending them to the LLM for a grounded answer.

---

## Project Structure

```
RAG-AI-Chatbot/
├── phase2.py              # Phase 2: Basic LLM chatbot (Streamlit + Groq)
├── phase3.py              # Phase 3: Full RAG chatbot (recommended)
├── OperatingSystems.pdf   # Sample knowledge-base document
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependency versions (uv)
├── .python-version        # Pinned Python version (3.10)
└── README.md
```

---

## Complete Project Flow

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                        │
│          User types a question in the chat box          │
└────────────────────────┬────────────────────────────────┘
                         │ user prompt
                         ▼
┌─────────────────────────────────────────────────────────┐
│               Session State (Chat History)              │
│   Stores all past messages so the UI stays persistent   │
└────────────────────────┬────────────────────────────────┘
                         │
           ┌─────────────▼──────────────┐
           │  Document Ingestion        │  (cached – runs once)
           │  ─────────────────────     │
           │  1. Load PDF               │
           │     PyPDFLoader            │
           │     (OperatingSystems.pdf) │
           │                            │
           │  2. Split into chunks      │
           │     RecursiveCharacter     │
           │     TextSplitter           │
           │     chunk_size  = 1000    │
           │     chunk_overlap = 100    │
           │                            │
           │  3. Embed chunks           │
           │     HuggingFaceEmbeddings  │
           │     (all-MiniLM-L12-v2)   │
           │                            │
           │  4. Store in Chroma DB     │
           │     (in-memory vector      │
           │      store)                │
           └─────────────┬──────────────┘
                         │
           ┌─────────────▼──────────────┐
           │  Retrieval Stage           │
           │  ─────────────────────     │
           │  Embed the user query      │
           │  Search Chroma DB for      │
           │  top-3 similar chunks      │
           └─────────────┬──────────────┘
                         │ retrieved context
           ┌─────────────▼──────────────┐
           │  Generation Stage          │
           │  ─────────────────────     │
           │  RetrievalQA chain         │
           │  (chain_type = "stuff")    │
           │                            │
           │  Combine context + query   │
           │  → send to Groq LLM        │
           │     (llama3-8b-8192)       │
           └─────────────┬──────────────┘
                         │ generated answer
                         ▼
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                        │
│    Answer displayed in chat; appended to history        │
└─────────────────────────────────────────────────────────┘
```

### Step-by-step description

1. **User sends a prompt** via the Streamlit chat input widget.
2. **Chat history** is maintained in `st.session_state` so all previous messages are shown on re-render.
3. **Document ingestion** (cached with `@st.cache_resource`):
   - The PDF is loaded page-by-page with `PyPDFLoader`.
   - The text is split into overlapping chunks (`chunk_size=1000`, `chunk_overlap=100`) using `RecursiveCharacterTextSplitter`.
   - Each chunk is converted into a dense vector with the `all-MiniLM-L12-v2` HuggingFace embedding model.
   - All vectors are stored in an in-memory **Chroma** vector store.
4. **Retrieval**: The user query is embedded with the same model, and the **top 3** most semantically similar chunks are retrieved from Chroma.
5. **Generation**: The retrieved chunks are concatenated ("stuffed") with the query and passed to **Groq's Llama 3 (8B)** model via LangChain's `RetrievalQA` chain.
6. **Response** is streamed back and displayed in the chat window; it is also saved to session history.

---

## Technologies Used

| Category | Library / Service | Version |
|----------|------------------|---------|
| Web UI | [Streamlit](https://streamlit.io) | ≥ 1.44.1 |
| LLM Orchestration | [LangChain](https://langchain.com) | ≥ 0.3.23 |
| LLM Provider | [Groq](https://console.groq.com) – Llama 3 8B | API |
| LangChain × Groq | langchain-groq | ≥ 0.3.2 |
| PDF Parsing | [PyPDF](https://pypdf.readthedocs.io) | ≥ 5.4.0 |
| Embeddings | [Sentence-Transformers](https://www.sbert.net) (`all-MiniLM-L12-v2`) | ≥ 2.2.2 |
| Vector Store | Chroma DB (via LangChain community) | built-in |
| Package Manager | [uv](https://github.com/astral-sh/uv) | — |
| Python | 3.10+ | — |

---

## Prerequisites

- Python **3.10** or newer
- A free **Groq API key** – sign up at <https://console.groq.com>
- `uv` (recommended) **or** `pip`

---

## Installation

### Option A – uv (recommended)

```bash
# Install uv if you don't have it
pip install uv

# Clone the repository
git clone https://github.com/unstoppableayush/RAG-AI-Chatbot.git
cd RAG-AI-Chatbot

# Create a virtual environment and install dependencies
uv sync
```

### Option B – pip

```bash
git clone https://github.com/unstoppableayush/RAG-AI-Chatbot.git
cd RAG-AI-Chatbot

pip install langchain langchain-community langchain-groq \
            pypdf sentence-transformers streamlit
```

---

## Configuration

The only required environment variable is your **Groq API key**:

```bash
export GROQ_API_KEY="gsk_your_key_here"
```

> **Tip:** Create a `.env` file in the project root and use a tool like `python-dotenv` to load it automatically, or simply export the variable in your shell profile.

---

## Usage

### Run the RAG Chatbot (Phase 3 – recommended)

```bash
streamlit run phase3.py
```

Open <http://localhost:8501> in your browser. The chatbot will load the `OperatingSystems.pdf` knowledge base on first launch (this is cached automatically), and you can start asking questions.

### Run the Basic LLM Chatbot (Phase 2)

```bash
streamlit run phase2.py
```

This version does **not** use document retrieval; it answers general questions using the Groq LLM directly.

---

## Phase Comparison

| Feature | Phase 2 (`phase2.py`) | Phase 3 (`phase3.py`) |
|---------|----------------------|----------------------|
| Streamlit UI | ✅ | ✅ |
| Chat history | ✅ | ✅ |
| Groq LLM | ✅ Llama 3 8B | ✅ Llama 3 8B |
| PDF loading | ❌ | ✅ PyPDFLoader |
| Text chunking | ❌ | ✅ RecursiveCharacterTextSplitter |
| Embeddings | ❌ | ✅ all-MiniLM-L12-v2 |
| Vector store | ❌ | ✅ Chroma DB |
| RAG retrieval | ❌ | ✅ Top-3 chunks |
| Error handling | ❌ | ✅ try / except |
| Response caching | ❌ | ✅ @st.cache_resource |
