import sys
import os

# Ensure we can import from the brain module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from brain.engine import Brain

class Cleaner:
    def __init__(self, brain_model="llama3.2", language="en"):
        """
        Initializes the Cleaner module, linking it to the local LLM brain.
        """
        self.brain = Brain(model=brain_model)
        self.language = language
        self.system_prompt = (
            f"Actúa como un corrector ortográfico de español. Tu única tarea es corregir la puntuación y ortografía del texto. "
            f"REGLAS CRÍTICAS:\n"
            f"1. No cambies las palabras. No cambies el sentido.\n"
            f"2. Si ves 'exceptico', corrígelo a 'escéptico'.\n"
            f"3. Mantén las etiquetas con corchetes [Nombre] y los tiempos exactamente igual.\n"
            f"4. No añadas notas, comentarios ni introducciones. Solo devuelve el texto corregido."
        )

    def process(self, raw_transcript):
        """
        Takes raw text and returns a cleaned version.
        """
        if not raw_transcript.strip():
            return ""

        print(f"Cleaning transcript ({len(raw_transcript)} characters)...")
        
        # We send the transcript directly to avoid LLM confusion with "Input/Output" labels
        cleaned_text = self.brain.query(raw_transcript, system_prompt=self.system_prompt)
        
        # Final safety: remove any "Nota:" or "Corrección:" if the LLM ignored the rules
        if "Nota:" in cleaned_text:
            cleaned_text = cleaned_text.split("Nota:")[0].strip()
        
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
