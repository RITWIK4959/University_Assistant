# Nexi — SRM University Assistant

Nexi is a voice-enabled campus assistant built with LiveKit agents + a RAG engine and local Chroma DB. It answers student questions conversationally and can use stored documents from the "Nexi Data" corpus.

## Quick summary
- Real-time audio agent using LiveKit (agents, STT, TTS, VAD).
- Retrieval-Augmented Generation via local RAG engine.
- Document store: Chroma DB at [chromadb/chroma.sqlite3](chromadb/chroma.sqlite3).
- Entry point: [agent_patched.py](agent_patched.py).

## Key files & symbols
- [`livekit_patch.apply_livekit_patch`](livekit_patch.py) — patch applied before importing LiveKit: [livekit_patch.py](livekit_patch.py)
- [`agent_patched.Assistant`](agent_patched.py) — Agent subclass and main orchestration: [agent_patched.py](agent_patched.py)
- [`agent_patched.entrypoint`](agent_patched.py) — worker entry: [agent_patched.py](agent_patched.py)
- [`agent_patched.clean_response_for_speech`](agent_patched.py) — sanitize RAG output: [agent_patched.py](agent_patched.py)
- [`rag_engine.initialize_rag_engine`](rag_engine.py) — RAG init: [rag_engine.py](rag_engine.py)
- [`rag_engine.get_rag_answer_async`](rag_engine.py) — async retrieval + answer: [rag_engine.py](rag_engine.py)
- Local datastore files:
  - [chromadb/chroma.sqlite3](chromadb/chroma.sqlite3)
  - [storage/docstore.json](storage/docstore.json)
  - [storage/index_store.json](storage/index_store.json)
  - [storage/image__vector_store.json](storage/image__vector_store.json)
  - [storage/graph_store.json](storage/graph_store.json)

## Setup (dev)
1. Create venv and activate:
   - python -m venv .venv && source .venv/bin/activate
2. Install:
   - pip install -r [requirements.txt](requirements.txt)
3. Add secrets:
   - Copy and edit `.env.local` (used by [agent_patched.py](agent_patched.py)).
4. Ensure your Chroma DB and "Nexi Data" corpus are in place:
   - [chromadb/chroma.sqlite3](chromadb/chroma.sqlite3)
   - [Nexi Data/](Nexi Data/)

## Run (development)
- Start the agent worker (this runs the `entrypoint`):
  - python agent_patched.py
- The app uses [`agent_patched.entrypoint`](agent_patched.py) and starts the LiveKit agent session and RAG flow.

## How RAG flow works
- RAG is initialized via [`rag_engine.initialize_rag_engine`](rag_engine.py).
- On a user query, the agent calls [`rag_engine.get_rag_answer_async`](rag_engine.py), sanitizes with [`agent_patched.clean_response_for_speech`](agent_patched.py), then sends instructions to the LLM.

## Notes & troubleshooting
- If LiveKit import issues occur, ensure [`livekit_patch.apply_livekit_patch`](livekit_patch.py) runs before LiveKit is imported.
- For missing embeddings or corrupt DB, inspect [chromadb/chroma.sqlite3](chromadb/chroma.sqlite3).
- Logs are controlled by Python logging in [agent_patched.py](agent_patched.py).

## Development tips
- Keep document updates in "Nexi Data/" and re-run your RAG ingestion pipeline (see [`rag_engine.py`](rag_engine.py)).
- Use the integrated LiveKit session callbacks in [agent_patched.py](agent_patched.py) to extend behavior (e.g., add events on transcription).
