# System Design: Personal LLM (Local RAG Architecture)

## 1. System Overview
PersonalLLM is an entirely offline, privacy-first "Second Brain" and conversational memory assistant. It utilizes a **Retrieval-Augmented Generation (RAG)** architecture to process audio input, store the semantic meaning of those conversations, and retrieve them later to answer user queries.

**Key Constraints & Principles:**
*   **100% Local Processing:** No data is sent to external APIs.
*   **Modularity:** The system is divided into decoupled components (Ears, Brain, Cleaner, Memory) to allow easy swapping of underlying technologies (e.g., swapping Whisper for another STT engine).
*   **Consumer-Hardware Friendly:** Default configurations use heavily quantized and small-parameter models (e.g., Llama 3.2 1B, Whisper Tiny) to run on standard CPUs/GPUs.

---

## 2. High-Level Architecture

The system operates on two distinct data pipelines: **Ingestion** and **Retrieval**.

```text
[ Microphone ] 
      │
      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  EARS        │───>  │  CLEANER     │───>  │  MEMORY      │───>  │  DISK        │
│ (Whisper)    │ Raw  │ (Ollama LLM) │Clean │ (ChromaDB)   │Vector│ (SQLite/   │
│ Speech-to-Tx │ Text │ Noise Filter │Text  │ Embeddings   │Data  │  Parquet)    │
└──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
                                                                        │
                                                                        ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  USER QUERY  │───>  │  MEMORY      │───>  │  BRAIN       │───>  │  FINAL       │
│ (Text Input) │      │ (ChromaDB)   │Top K │ (Ollama LLM) │      │  ANSWER      │
└──────────────┘      │ Vector Search│Docs  │ RAG Prompt   │      └──────────────┘
                      └──────────────┘      └──────────────┘
```

---

## 3. Component Breakdown

### 3.1. The Brain (`brain/engine.py`)
*   **Role:** The core reasoning engine. It acts as the intelligent processor for both the *Cleaner* (summarizing) and the *Orchestrator* (answering questions).
*   **Technology:** [Ollama](https://ollama.com/) running `llama3.2:1b` (or higher).
*   **Design Decision:** Ollama abstracts away the complexities of managing PyTorch/llama.cpp inference and provides a clean HTTP API (`http://localhost:11434/api/generate`). We use the `1b` parameter model by default for near-instant inference, though it trades off some complex reasoning capabilities.

### 3.2. The Ears (`ears/transcriber.py` & `ears/recorder.py`)
*   **Role:** Converts human speech into raw text.
*   **Technology:** `sounddevice` (audio capture) + `openai-whisper` (Speech-to-Text).
*   **Design Decision:** We use Whisper's `tiny` model. Whisper uses a Transformer sequence-to-sequence architecture. The raw transcript is often messy, containing filler words ("um", "ah") and poor punctuation, which necessitates the Cleaner module.

### 3.3. The Cleaner (`cleaner/processor.py`)
*   **Role:** An intermediary processing step (Data Transformation). It takes the raw output from the Ears and normalizes it.
*   **Technology:** Ollama (Brain) via a strict System Prompt.
*   **Design Decision:** Storing raw conversational text directly into a Vector DB degrades search quality because filler words and conversational tangents create "noise" in the vector embeddings. By summarizing and structuring the text first, we significantly improve the accuracy of the semantic search later.

### 3.4. The Memory (`memory/storage.py`)
*   **Role:** Long-term persistence and semantic retrieval.
*   **Technology:** [ChromaDB](https://www.trychroma.com/).
*   **How it works (Embeddings):** When text is passed to ChromaDB, it runs it through an embedding model (default: `all-MiniLM-L6-v2`). This model converts the text into a dense vector (a list of floating-point numbers) representing the *semantic meaning* of the sentence. 
*   **How it works (Retrieval):** When a user asks a question, the query is also converted into a vector. ChromaDB calculates the cosine similarity (distance) between the query vector and all stored document vectors, returning the closest matches.
*   **Metadata:** We attach UNIX timestamps and source identifiers to enable time-based filtering and data expiration (TTL logic).

---

## 4. Data Flow Pipelines

### A. The Ingestion Pipeline (`app.ingest_audio`)
1.  **Capture:** User speaks into the microphone; saved to `user_recording.wav`.
2.  **Transcription:** Whisper processes the `.wav` file -> returns `raw_text` string.
3.  **Sanitization:** Cleaner wraps `raw_text` in a prompt instructing the LLM to remove filler words. Returns `clean_text`.
4.  **Vectorization & Storage:** `clean_text` is sent to ChromaDB, embedded into a vector, and saved to the local SQLite database along with `{ "timestamp": 1711929384 }`.

### B. The RAG Pipeline (`app.ask`)
1.  **Querying:** User inputs a question: *"What is the wifi password?"*
2.  **Vector Search:** ChromaDB embeds the question and searches the database for the `Top K` (e.g., 2) most mathematically similar saved memories.
3.  **Prompt Construction:** The retrieved memories (Context) are dynamically injected into a Master Prompt alongside the user's Question.
4.  **Generation:** The Brain LLM reads the Master Prompt. Because the answer is contained within the injected context, it avoids hallucination and generates an accurate response based *only* on the provided memories.

---

## 5. Trade-offs and Tuning (Developer Notes)

If you wish to scale or modify this system, consider the following trade-offs:

| Component | Current Setup | Pros | Cons | Upgrade Path |
| :--- | :--- | :--- | :--- | :--- |
| **LLM (Brain)** | `llama3.2:1b` | Lightning fast, low RAM/VRAM usage. | Struggles with complex RAG instructions; chatty. | Change to `llama3.2` (3B) or `mistral` (7B) in `main.py`. |
| **STT (Ears)** | Whisper `tiny` | Very fast on CPU (~70MB). | Poor accuracy with accents or background noise. | Change `model_size="tiny"` to `"base"` or `"small"` in `main.py`. |
| **Vector DB** | ChromaDB (SQLite) | Zero setup, runs entirely locally. | Not meant for massive enterprise concurrency. | Migrate to Milvus or Qdrant for massive scale. |
| **Embeddings** | `all-MiniLM-L6-v2` | Default Chroma model, fast and small. | Context window limited to 256 tokens per chunk. | Use `nomic-embed-text` via Ollama for larger context embeddings. |

## 6. Future Expansion Ideas
*   **Speaker Diarization:** Update the Ears module using `pyannote.audio` to identify *who* is speaking (e.g., "User vs. Guest").
*   **Continuous Listening:** Implement Voice Activity Detection (VAD) via `webrtcvad` to allow the system to listen in the background and only trigger Whisper when someone is actually speaking.
*   **Memory Paging:** Implement an active "Working Memory" vs "Long Term Memory" system (similar to the Letta/MemGPT architecture) for infinite context.