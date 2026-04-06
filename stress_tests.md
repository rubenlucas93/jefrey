# Personal LLM - Stress Tests & Evolution Guide

This document contains a checklist of extreme real-world scenarios to test the limits of the Personal LLM. 
Go through these one by one. If a test fails, use the "How to fix" strategies to evolve the codebase.

## 1. 🌊 The Background Noise Test (Whisper & VAD)
*   **Task:** Record a 30-second conversation in a noisy environment (e.g., kitchen with running water, or with a TV playing in the background). Ingest it and ask a question about what you said.
*   **Goal:** The system correctly transcribes only your voices and ignores or filters out the TV/noise.
*   **If Failure:**
    *   *Issue:* Whisper hallucinated TV dialogue, or VAD kept recording silence/noise.
    *   *Fix:* Upgrade `whisper_model` to `"medium"` in `config.json`. 
    *   *Fix:* Implement stricter Voice Activity Detection (VAD) thresholds in `ears/recorder.py` to drop audio chunks that lack clear human speech.

## 2. 🗣️ The Interruption Test (Diarization)
*   **Task:** Record an audio clip where Rubén and Cristina interrupt each other and talk at the exact same time. Ask a question about who said the interrupted word.
*   **Goal:** Pyannote successfully separates the overlapping audio into distinct `[Rubén]` and `[Cristina]` tags with correct timestamps.
*   **If Failure:**
    *   *Issue:* Pyannote merges both voices into one tag or assigns the wrong speaker.
    *   *Fix:* Adjust the diarization parameters in `ears/diarizer.py`. You may need to look into Pyannote's `onset` and `offset` thresholds, or rely on the LLM in the `Cleaner` step to logically separate two sentences mashed together.

## 3. 👤 The Guest Test (Biometrics Thresholds)
*   **Task:** Have a third person (a friend or family member) speak into the microphone for a few sentences.
*   **Goal:** The system should NOT tag them as Rubén or Cristina. It should leave them as `[Speaker X]` or prompt you in the terminal with `¿Quién dijo esto?` to enroll them.
*   **If Failure:**
    *   *Issue:* The guest is falsely identified as Rubén or Cristina.
    *   *Fix:* The biometric similarity threshold is too loose. Open `ears/biometrics.py` and increase the `threshold` in `identify_speaker()` (e.g., from `0.65` to `0.75` or `0.80`).

## 4. 🔄 The Changing Facts Test (Memory & RAG)
*   **Task:** 
    1. Record: "He dejado las llaves en la mesa del salón." (Ingest it).
    2. Wait 1 minute.
    3. Record: "Oye, he movido las llaves al cajón de la cocina." (Ingest it).
    4. Ask: "¿Dónde están las llaves?"
*   **Goal:** The Brain answers "En el cajón de la cocina" because it recognizes it as the most recent fact.
*   **If Failure:**
    *   *Issue:* The LLM gets confused and says "en la mesa" or combines both locations.
    *   *Fix:* Update the RAG prompt in `main.py` to strongly emphasize that higher timestamps overwrite older ones. 
    *   *Fix:* Modify `memory/storage.py` to strictly sort the vector search results by the `timestamp` metadata before passing them to the LLM.

## 5. ⏳ The Temporal Reasoning Test (LLM Logic)
*   **Task:** Wait a day (or simulate it by manually editing a timestamp in the database). Ask the system: "¿Qué dije que iba a hacer ayer?"
*   **Goal:** The LLM does the math between `FECHA ACTUAL` and the memory's timestamp, correctly identifying the memory from the previous day.
*   **If Failure:**
    *   *Issue:* The 3B model fails at date math and says there is no memory.
    *   *Fix:* Add a pre-processing step in `main.py` before querying the LLM that translates timestamps into human terms (e.g., `[Ayer]`, `[Hace 2 horas]`) so the LLM doesn't have to calculate dates mentally.

## 6. 🗣️ The "Umm... Ahh..." Test (Cleaner Reboot)
*   **Task:** Speak very naturally, with stutters, false starts, and filler words: *"Yo, ehh, creo que... no sé, igual dejé el... el cargador en el coche."* Ask where the charger is.
*   **Goal:** The system understands the core fact despite the messy text.
*   **If Failure:**
    *   *Issue:* The raw text is too messy for the RAG prompt to extract facts cleanly.
    *   *Fix:* Re-enable the LLM-based `Cleaner` in `cleaner/processor.py`. To prevent it from destroying the `[Name]` tags (which was happening in earlier iterations), give it a flawless few-shot prompt with strict JSON output, or consider using a larger model (like Llama 3 8B) exclusively for the cleaning step.

---
### Execution Log
Use this section to track your testing:

- [ ] Background Noise Test: (Result: ...)
- [ ] Interruption Test: (Result: ...)
- [ ] Guest Test: (Result: ...)
- [ ] Changing Facts Test: (Result: ...)
- [ ] Temporal Reasoning Test: (Result: ...)
- [ ] "Umm... Ahh..." Test: (Result: ...)
