# AI Agent

A simple conversational AI agent built with [LangGraph](https://langchain-ai.github.io/langgraph/) and Google's Gemini model.

## What this project does (so far)

Right now the project is a basic chatbot that runs in your terminal:

- You type a message.
- It's sent to Google's Gemini AI model, along with a short set of instructions (a "system prompt") telling the AI how to behave — be clear, admit when it doesn't know something, ask for clarification when needed, etc.
- The AI's reply is shown back to you, and the conversation keeps going until you type `quit` or `exit`.

Under the hood, this is built using **LangGraph**, a library for organizing an AI agent's logic as a flowchart (a "graph") of steps ("nodes") connected by arrows ("edges"). Right now the flowchart is intentionally simple — one step in, one step out — but it's structured so more steps can be added later. See the comments in [agent.py](agent.py) for a plain-language explanation of these concepts (graph, state, node, etc.).

## Project structure

- [agent.py](agent.py) — the chatbot agent: builds the LangGraph flowchart and runs the terminal chat loop.
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
3. Run the agent:
   ```
   python agent.py
   ```

## What's coming next

The project already includes dependencies for document loading and search (`pypdf`, `langchain-text-splitters`, `faiss-cpu`), which will be used to add a **knowledge base**: the ability for the agent to look up information from your own documents instead of relying only on what the AI model already knows. This is currently in progress.
