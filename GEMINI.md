# Project Mandates - Personal LLM

## Architectural Principles
- **Privacy First**: Everything must run locally. No external API calls for processing data.
- **Modularity**: Each component (Brain, Ears, Cleaner, Memory) must be independent and testable.
- **Efficiency**: Use quantized models (GGUF) via Ollama to ensure performance on consumer hardware.

## Progress Log
- [2026-03-31] Project initialized. SOTA research completed.
- [2026-03-31] Documentation created (README.md, GEMINI.md).
- [2026-03-31] Step 1 complete: Brain Module (Ollama with llama3.2:1b) configured and tested successfully.
- [2026-03-31] Step 2 complete: Ears Module (openai-whisper) installed and tested offline.
- [2026-03-31] Step 3 complete: Cleaner Module implemented using the local Ollama brain.
- [2026-03-31] Step 4 complete: Memory Module (ChromaDB) successfully tested with semantic search.
- [2026-03-31] Step 5 complete: Orchestrator created, tying Audio -> Whisper -> Llama -> ChromaDB -> RAG pipeline completely offline. Proof of concept successful!
