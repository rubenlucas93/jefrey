import os

# Prevent ChromaDB SQLite errors
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from brain.engine import Brain
from ears.transcriber import Ears
from cleaner.processor import Cleaner
from memory.storage import Memory

class PersonalLLM:
    def __init__(self):
        print("\n--- Initializing Personal LLM ---")
        self.brain = Brain(model="llama3.2:1b")
        self.ears = Ears(model_size="tiny")
        self.cleaner = Cleaner(brain_model="llama3.2:1b")
        self.memory = Memory()
        print("--- System Ready ---\n")

    def ingest_audio(self, audio_path):
        """
        Full pipeline: Audio -> Raw Text -> Clean Text -> Vector Memory
        """
        print(f"\n>>> 1. Hearing '{audio_path}'...")
        raw_text = self.ears.transcribe(audio_path)
        if not raw_text or "Error" in raw_text:
            print("No speech found or file error.")
            return
            
        print(f"\n🔍 [DEBUG - RAW TRANSCRIPT]:\n{raw_text}\n")

        print(">>> 2. Thinking & Cleaning...")
        cleaned_text = self.cleaner.process(raw_text)
        
        print(f"\n🧹 [DEBUG - CLEANED TRANSCRIPT]:\n{cleaned_text}\n")
        
        print(">>> 3. Remembering...")
        doc_id = self.memory.remember(cleaned_text, metadata={"source": audio_path})
        print(f"Stored securely offline (ID: {doc_id})")

    def ask(self, question):
        """
        Full pipeline: Question -> Recall Context -> Generate Answer
        """
        print(f"\n>>> Question: {question}")
        print("Recalling from memory...")
        
        # Pull the top 2 most relevant past conversation summaries
        context_docs = self.memory.recall(question, n_results=2)
        
        if not context_docs:
            context = "No specific memories found about this."
            print(">> No memories recalled.")
        else:
            context = " ".join(context_docs)
            print(f">> Recalled Context: '{context}'")

        print("Consulting the Brain...")
        
        # Build a prompt using the recalled memories (RAG)
        prompt = (
            f"CONTEXT: '{context}'\n"
            f"QUESTION: {question}\n\n"
            f"Use the CONTEXT above to answer. If a name is mentioned in the context, use that EXACT name. "
            f"Do not guess who the person is. Answer in one short sentence."
        )
        
        answer = self.brain.query(prompt, system_prompt="You are a precise and helpful assistant.")
        print(f"\n🤖 Answer:\n{answer}\n")
if __name__ == "__main__":
    from ears.recorder import record_audio

    app = PersonalLLM()

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
