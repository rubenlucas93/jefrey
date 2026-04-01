import sys
import os

# Ensure we can import from the brain module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from brain.engine import Brain

class Cleaner:
    def __init__(self, brain_model="llama3.2"):
        """
        Initializes the Cleaner module, linking it to the local LLM brain.
        """
        self.brain = Brain(model=brain_model)
        self.system_prompt = (
            "You are a transcription cleaner. Your job is to take raw, messy "
            "spoken text and convert it into a clean, highly readable summary. "
            "Remove all filler words ('um', 'ah', 'like', 'you know'). "
            "Fix grammar but preserve the original meaning and core points exactly. "
            "Do not add any conversational preamble like 'Here is the summary:'."
        )

    def process(self, raw_transcript):
        """
        Takes raw text and returns a cleaned version.
        """
        if not raw_transcript.strip():
            return ""

        print(f"Cleaning transcript ({len(raw_transcript)} characters)...")
        
        prompt = f"Please clean and summarize the following transcript:\n\n{raw_transcript}"
        cleaned_text = self.brain.query(prompt, system_prompt=self.system_prompt)
        
        return cleaned_text

if __name__ == "__main__":
    cleaner = Cleaner()
    
    noisy_text = (
        "Um, yeah, so I was thinking about, like, downloading an LLM, you know? "
        "And maybe making it work offline. Ah, then transcription of, like, all my conversations. "
        "And then, I guess, storing them so I can, um, ask it for advice later?"
    )
    
    print("--- RAW TEXT ---")
    print(noisy_text)
    
    cleaned = cleaner.process(noisy_text)
    
    print("\n--- CLEANED TEXT ---")
    print(cleaned)
    print("--------------------")
