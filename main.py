import os
import argparse
import logging
import json

# Prevent ChromaDB SQLite errors
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from brain.engine import Brain
from ears.transcriber import Ears
from ears.diarizer import Diarizer
from ears.biometrics import VoiceBiometrics
from cleaner.processor import Cleaner
from memory.storage import Memory

class PersonalLLM:
    def __init__(self, debug=False):
        self.debug = debug
        
        # Load configuration
        self.config = {"language": "es", "whisper_model": "small", "llm_model": "llama3.2:latest"}
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    self.config.update(json.load(f))
            except Exception as e:
                print(f"Warning: Could not read config.json: {e}")

        self.language = self.config.get("language", "en")
        llm_model = self.config.get("llm_model", "llama3.2:latest")
        whisper_model = self.config.get("whisper_model", "small")
        self.hf_token = self.config.get("hf_token", "")

        if self.debug:
            print(f"\n[DEBUG] --- Initializing Personal LLM ({self.language}) in DEBUG mode ---")
        else:
            print(f"\n--- Initializing Personal LLM ({self.language}) ---")
            
        self.brain = Brain(model=llm_model)
        self.ears = Ears(model_size=whisper_model)
        self.diarizer = Diarizer(hf_token=self.hf_token, language=self.language)
        
        # Initialize Voice Biometrics if diarizer pipeline loaded successfully
        self.biometrics = None
        if self.diarizer.pipeline:
            self.biometrics = VoiceBiometrics(self.diarizer.pipeline)
            
        self.cleaner = Cleaner(brain_model=llm_model, language=self.language)
        self.memory = Memory()
        print("--- System Ready ---\n")

    def ingest_audio(self, audio_path):
        """
        Full pipeline: Audio -> Raw Text -> Speaker Tagged -> Clean Text -> Vector Memory
        """
        print(f"\n>>> 1. Hearing '{audio_path}' (Language: {self.language})...")
        result = self.ears.transcribe(audio_path, language=self.language)
        raw_text = result.get("text", "")
        segments = result.get("segments", [])

        if not raw_text or "Error" in raw_text:
            print("No speech found or file error.")
            return
            
        if self.debug:
            print(f"\n🔍 [DEBUG - RAW TRANSCRIPT]:\n{raw_text}\n")

        print(">>> 2. Identifying Speakers (Audio Analysis)...")
        tagged_text = self.diarizer.tag_speakers(audio_path, segments)
        
        import re
        unique_speakers = set(re.findall(r'\[(Speaker \d+)\]', tagged_text))
        if unique_speakers:
            mapping = {}
            for speaker in sorted(unique_speakers):
                snippet = ""
                start_time, end_time = 0.0, 0.0
                
                for line in tagged_text.split('\n'):
                    if line.startswith(f"[{speaker}]"):
                        # Extract timestamps like [Speaker 1] 0.0-5.0:
                        try:
                            time_match = re.search(r'(\d+\.\d+)-(\d+\.\d+):', line)
                            if time_match:
                                start_time = float(time_match.group(1))
                                end_time = float(time_match.group(2))
                            snippet = line.split(':', 1)[1].strip()
                        except:
                            snippet = line
                        break
                
                # Biometric extraction & identification
                speaker_emb = None
                auto_identified = None
                if self.biometrics and (end_time - start_time) >= 0.5:
                    speaker_emb = self.biometrics.extract_embedding(audio_path, start_time, end_time)
                    auto_identified = self.biometrics.identify_speaker(speaker_emb)
                
                if auto_identified:
                    mapping[speaker] = auto_identified
                elif not getattr(self, "skip_mapping", False):
                    # Ask the user who this speaker is
                    print(f"\n👥 Identificando voz: {speaker}")
                    user_input = input(f"¿Quién dijo esto? \"{snippet[:60]}...\" (Enter para dejar como '{speaker}'): ")
                    if user_input.strip():
                        mapping[speaker] = user_input.strip()
                        # Enroll the new speaker to the DB
                        if self.biometrics and speaker_emb is not None:
                            self.biometrics.enroll_speaker(mapping[speaker], speaker_emb)
            
            # Replace tags with actual names
            for speaker, name in mapping.items():
                tagged_text = tagged_text.replace(f"[{speaker}]", f"[{name}]")
        
        if self.debug:
            print(f"\n👥 [DEBUG - TAGGED TRANSCRIPT]:\n{tagged_text}\n")

        print(">>> 3. Thinking & Cleaning...")
        # We pass the tagged text to the cleaner so it keeps the speaker labels
        cleaned_text = self.cleaner.process(tagged_text)
        
        if self.debug:
            print(f"\n🧹 [DEBUG - CLEANED TRANSCRIPT]:\n{cleaned_text}\n")
        
        print(">>> 4. Remembering...")
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamped_text = f"[{now}]\n{cleaned_text}"
        doc_id = self.memory.remember(timestamped_text, metadata={"source": audio_path, "timestamp": now})
        print(f"Stored securely offline (ID: {doc_id})")

    def ask(self, question):
        """
        Full pipeline: Question -> Recall Context -> Generate Answer
        """
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n>>> Question: {question}")
        
        if self.debug:
            print("[DEBUG] Recalling from memory...")
        
        # Pull the top 5 most relevant past conversation summaries
        context_docs = self.memory.recall(question, n_results=5)
        
        if not context_docs:
            context = "No se encontraron memorias sobre esto."
            if self.debug:
                print("[DEBUG] No memories recalled.")
        else:
            context = "\n".join(context_docs)
            if self.debug:
                print(f"[DEBUG] Recalled Context:\n{context}")

        if self.debug:
            print("[DEBUG] Consulting the Brain...")
        
        # Build a prompt using the recalled memories (RAG)
        prompt = (
            f"FECHA ACTUAL: {now}\n\n"
            f"MEMORIAS:\n"
            f"```\n{context}\n```\n\n"
            f"REGLAS DE LÓGICA:\n"
            f"- Si [Cristina] dice 'tú escogiste', significa que RUBÉN escogió.\n"
            f"- Si [Rubén] dice 'tú escogiste', significa que CRISTINA escogió.\n"
            f"- Verbos terminados en '-aste' o '-iste' se refieren a la OTRA persona.\n"
            f"- Verbos terminados en '-é' o '-í' se refieren a la misma persona que habla.\n\n"
            f"PREGUNTA: {question}\n"
            f"RESPUESTA (Solo el nombre o el dato):"
        )
        
        answer = self.brain.query(prompt, system_prompt="Eres un asistente personal preciso que actúa como memoria para el usuario.")
        print(f"\n🤖 Answer:\n{answer}\n")
        return answer

if __name__ == "__main__":
    from ears.recorder import record_audio

    parser = argparse.ArgumentParser(description="Personal LLM Orchestrator")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output")
    parser.add_argument("--file", type=str, help="Path to an existing audio file to process instead of recording")
    parser.add_argument("--ambient", action="store_true", help="Run continuously in ambient mode (background recording + foreground querying)")
    args = parser.parse_args()

    app = PersonalLLM(debug=args.debug)

    if args.ambient:
        import queue
        import threading
        from ears.recorder import ambient_recorder_thread
        from memory.retention import prune_old_audio
        
        # In ambient mode, we don't want the app to prompt for 'Quién dijo esto?' and block the queue
        app.skip_mapping = True 

        audio_queue = queue.Queue()
        
        # Start the background recorder (Records 30s chunks)
        rec_thread = threading.Thread(target=ambient_recorder_thread, args=(audio_queue,), daemon=True)
        rec_thread.start()
        
        # Start the background processor (Transcribes and saves to ChromaDB)
        def process_worker():
            while True:
                filename = audio_queue.get()
                try:
                    prune_old_audio()  # Auto-delete audio older than 48 hours to save space
                    app.ingest_audio(filename)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                audio_queue.task_done()
                
        proc_thread = threading.Thread(target=process_worker, daemon=True)
        proc_thread.start()
        
        print("\n" + "="*50)
        print("🏠 AMBIENT MODE ACTIVE")
        print("The system is listening and processing in the background.")
        print("You can ask questions at any time below.")
        print("="*50)
        
        while True:
            try:
                question = input("\nAsk the Brain (or type 'quit'): ")
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if question.strip():
                    app.ask(question)
            except KeyboardInterrupt:
                break
            except EOFError:
                break

    elif args.file:
        audio_file = args.file
        if not os.path.exists(audio_file):
            print(f"Error: File '{audio_file}' not found.")
            sys.exit(1)
        print(f"\n[Using existing audio file: {audio_file}]")
        result = audio_file
    else:
        # 1. Start live microphone recording
        audio_file = "user_recording.wav"
        print("\n[Press Ctrl+C to stop recording when you are done speaking]")
        result = record_audio(audio_file)

    # 2. Ingest the audio file if the user recorded something successfully
    if result:
        app.ingest_audio(audio_file)

        # 3. Allow the user to ask a question interactively via the terminal
        print("\n" + "="*50)
        print("🤔 Memory ingested! Test the brain:")
        while True:
            try:
                question = input("\nAsk a question about what you just said (or type 'quit'): ")
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                app.ask(question)
            except KeyboardInterrupt:
                break
            except EOFError:
                break
    else:
        print("Recording failed. Please check your microphone.")
