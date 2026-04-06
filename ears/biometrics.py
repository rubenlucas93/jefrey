import os
import json
import numpy as np
import torch
from pyannote.core import Segment
from pyannote.audio.core.io import Audio

class VoiceBiometrics:
    def __init__(self, pipeline, db_path="speaker_embeddings.json"):
        self.pipeline = pipeline
        self.db_path = db_path
        self.audio = Audio(sample_rate=16000, mono=True)
        self.embeddings = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                    # Convert lists back to numpy arrays
                    return {name: np.array(vec) for name, vec in data.items()}
            except Exception as e:
                print(f"Warning: Could not load voice biometrics DB: {e}")
        return {}

    def _save_db(self):
        try:
            with open(self.db_path, "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                data = {name: vec.tolist() for name, vec in self.embeddings.items()}
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not save voice biometrics DB: {e}")

    def extract_embedding(self, audio_path, start_time, end_time):
        """Extracts a 256-d voice print for a specific time segment."""
        try:
            # Ensure the segment is at least 0.5s for a stable embedding
            duration = end_time - start_time
            if duration < 0.5:
                return None
                
            waveform, _ = self.audio.crop(audio_path, Segment(start_time, end_time))
            # The _embedding model expects a batch dimension
            with torch.no_grad():
                emb = self.pipeline._embedding(waveform[None])
            return emb[0]
        except Exception as e:
            print(f"Warning: Could not extract embedding: {e}")
            return None

    def enroll_speaker(self, name, embedding):
        """Saves or updates a speaker's voice print."""
        if embedding is None:
            return
            
        if name in self.embeddings:
            # Moving average to slowly adapt the voice profile over time
            self.embeddings[name] = 0.8 * self.embeddings[name] + 0.2 * embedding
        else:
            self.embeddings[name] = embedding
            
        # Normalize to unit length
        self.embeddings[name] = self.embeddings[name] / np.linalg.norm(self.embeddings[name])
        self._save_db()
        print(f"[Biometrics] Voice print saved for '{name}'.")

    def identify_speaker(self, embedding, threshold=0.65):
        """Compares the embedding to the DB. Returns the name if a match is found."""
        if embedding is None or not self.embeddings:
            return None
            
        best_name = None
        best_score = -1.0
        
        # Normalize input
        emb_norm = embedding / np.linalg.norm(embedding)
        
        for name, known_emb in self.embeddings.items():
            # Cosine similarity
            score = np.dot(emb_norm, known_emb)
            if score > best_score:
                best_score = score
                best_name = name
                
        if best_score >= threshold:
            print(f"[Biometrics] Auto-identified '{best_name}' (Confidence: {best_score:.2f})")
            return best_name
            
        return None
