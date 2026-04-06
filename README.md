# Personal LLM (Offline Conversation Memory)

A private, offline "second brain" that transcribes, cleans, and stores your conversations to provide long-term memory and advice.

## Project Status: Step-by-Step Build

1.  [x] **Brain Module**: LLM Engine (Ollama integration, `llama3.2:latest`).
2.  [x] **Ears Module**: Transcription (`openai-whisper` base/small) and Speaker Diarization (`pyannote.audio` 3.1).
3.  [x] **Cleaner Module**: Text processing, Spanish spellchecking, and formatting via local LLM.
4.  [x] **Memory Module**: Vector storage and time-aware retrieval (`ChromaDB`).
5.  [x] **Orchestrator**: The central system connecting all modules.
6.  [x] **Voice Biometrics**: Automated cross-clip speaker recognition using voice prints (embeddings) to consistently map voices to names (e.g., Rubén, Cristina).

## Tech Stack (Current)
- **LLM**: [Ollama](https://ollama.com/) (Llama 3.2 3B)
- **Transcription**: [openai-whisper](https://github.com/openai/whisper)
- **Diarization**: [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- **Vector DB**: [ChromaDB](https://docs.trychroma.com/) (Memory)
- **Language**: Python 3.10+
- **Configuration**: `config.json` for language, models, and HF token.
