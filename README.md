# Personal LLM (Offline Conversation Memory)

A private, offline "second brain" that transcribes, cleans, and stores your conversations to provide long-term memory and advice.

## Project Status: Step-by-Step Build

1.  [x] **Brain Module**: LLM Engine (Ollama integration).
2.  [x] **Ears Module**: Transcription (openai-whisper).
3.  [x] **Cleaner Module**: Text processing and noise reduction.
4.  [x] **Memory Module**: Vector storage and retrieval.
5.  [x] **Orchestrator**: The central system connecting all modules.

## Tech Stack (Planned)
- **LLM**: [Ollama](https://ollama.com/) (Llama 3.2 / Phi-3.5)
- **Transcription**: [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- **Vector DB**: ChromaDB or LanceDB (Memory)
- **Language**: Python 3.10+
