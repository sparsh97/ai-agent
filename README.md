# AI Agent (Learning Project)

> 📚 **This is a personal learning project**, built to learn how AI agents, LangGraph, and Retrieval-Augmented Generation (RAG) work. It's not production software — code, structure, and docs are all still evolving as I learn.

A conversational AI agent built with [LangGraph](https://langchain-ai.github.io/langgraph/) and Google's Gemini model, with support for answering questions from your own documents (a "knowledge base").

## What this project does (so far)

There are two agents and a knowledge base pipeline:

### 1. `agent.py` — basic chatbot
A simple terminal chatbot: you type a message, it's sent to Gemini along with a system prompt telling it how to behave (be clear, admit when it doesn't know something, etc.), and the reply is printed back. Conversation continues until you type `quit` or `exit`.

### 2. `knowledge_base.py` — build a searchable knowledge base from a PDF
This is a one-time (or run-when-updated) script that prepares a document so an AI agent can search it later. It:

1. **Loads** a PDF (`vpn_user_guide.pdf`) into a list of documents (one per page).
2. **Splits** each document into small **chunks** of text (~1000 characters, with some overlap so ideas don't get cut in half).
3. **Embeds** each chunk — turns it into a list of numbers (an "embedding") that captures its meaning, using Google's `models/gemini-embedding-001` model — so chunks can be searched by meaning, not just keyword matching.
4. **Stores** all the chunks and their embeddings in a **FAISS vector store**, saved locally to the `faiss_index/` folder, so it can be searched instantly without redoing the above steps every time.

See the comments in [knowledge_base.py](knowledge_base.py) for a plain-language explanation of documents, chunks, embeddings, and vector stores.

### 3. `tech_support.py` — VPN support agent that uses the knowledge base
A more advanced version of the chatbot, built for Acme Corp IT support. Instead of only relying on what the AI model already knows, it can **search the knowledge base** (the `faiss_index/` built by `knowledge_base.py`) for real answers from the VPN user guide. It:

- Loads the saved `faiss_index/` vector store and wraps it as a **retriever tool** (`search_vpn_knowledge_base`) the AI can call.
- Uses a LangGraph flowchart with an extra step: the chatbot node can decide to call the search tool, get results back, and then reply — instead of always answering in one shot. This loop (`chatbot -> tools -> chatbot`) is what lets the agent "look things up" before responding.
- Only answers VPN-related questions, per its system prompt.

## Project structure

- [agent.py](agent.py) — basic chatbot: builds a simple LangGraph flowchart and runs the terminal chat loop.
- [tech_support.py](tech_support.py) — VPN support agent that searches the knowledge base via a retriever tool before answering.
- [knowledge_base.py](knowledge_base.py) — builds the searchable knowledge base (`faiss_index/`) from a PDF document.
- [src/ai_agent/](src/ai_agent/) — the installable Python package for this project (currently a placeholder, to be built out).
- [pyproject.toml](pyproject.toml) — project dependencies, managed with [uv](https://github.com/astral-sh/uv).

## Setup

1. Install dependencies:
   ```
   uv sync
   ```
2. Create a `.env` file in the project root with your Google API key:
   ```
   GOOGLE_API_KEY=your-key-here
   ```
3. Run the basic chatbot:
   ```
   uv run agent.py
   ```
4. To use the VPN support agent, first build the knowledge base (place a `vpn_user_guide.pdf` in the project root), then run the agent:
   ```
   uv run knowledge_base.py
   uv run tech_support.py
   ```

## What's coming next

Further knowledge-base improvements (e.g. supporting more documents, refining retrieval) are expected as this project continues to grow.
