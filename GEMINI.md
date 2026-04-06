# Autonomous Directive: Personal LLM (Offline Conversation Memory & Referee)

## 1. Context
This is an entirely offline, privacy-first "Second Brain" and Conversation Referee designed for a couple (Rubén and Cristina). It uses `openai-whisper` (transcription), `pyannote.audio` (speaker diarization + biometrics), `ChromaDB` (vector memory), and `Ollama` (Llama 3.2 for text cleaning and RAG Q&A). The primary target language is Spanish.

## 2. The Goal
Create a system that records conversations from a microphone all the time, drops periods of silence to save disk space and memory (Voice Activity Detection), and accurately answers questions like:
- "Where did I leave the keys?"
- "What did Cristina tell me yesterday?"
- "Who chose more movies this week?"
- "Who was right about what would happen to our friends?"
- "Who cleaned the most last week?"

**Success Metrics:**
Achieve a **>95% Success Score** across the pipeline during evaluation:
1.  **Transcription Accuracy:** Word Error Rate (WER) < 5%.
2.  **Speaker Identification:** >95% correct assignment of "Rubén" and "Cristina" across different clips using Biometrics.
3.  **Memory RAG:** >95% accuracy in answering contextual referee questions (zero hallucinations).

## 3. Technical Constraints
- Must use Python 3.10 inside the `venv`.
- All processing must remain 100% offline (no external APIs).
- Models: Whisper (small/medium), Pyannote 3.1, Llama 3.2 3B.
- Language: Spanish (`es`).
- **CRITICAL: NEVER STAGE OR COMMIT ANYTHING TO GIT. The user will manually review and push changes.**
- **CRITICAL:** Before writing massive features, check system health (sensors for heat, `df -h` for space). The system must warn the user if disk space is running low due to continuous recording.

## 4. Behavioral Rules (For the Autonomous Agent)
- **You are an autonomous researcher.** Before changing code, run `eval_loop.py` to establish a baseline.
- If a test fails or accuracy is below 95%, read the error/output, search for the fix (e.g., adjust `config.json` parameters, tweak LLM prompts in `cleaner/processor.py` or `main.py`, adjust biometric thresholds), and retry autonomously.
- Do NOT ask for permission to run tests, modify prompts, or tweak configurations. Iterate until the goal is met or you hit a hard technical wall.
- **Always log your progress, metric changes, and fixes to `AUTONOMOUS_LOG.txt`**.

## 5. The Execution Loop
1. Run `python eval_loop.py`.
2. Analyze the metrics (Transcription Score, Diarization Score, Q&A Score).
3. Identify the weakest link (e.g., LLM is hallucinating -> tighten prompt. Whisper is failing -> change model size. Pyannote is splitting voices -> adjust biometric threshold. Storage is full -> implement VAD chunking).
4. Apply the fix.
5. Record the new accuracy in `AUTONOMOUS_LOG.txt`.
6. Repeat until the overall score is >= 0.95.
