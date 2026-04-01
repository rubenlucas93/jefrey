import whisper
import os
import warnings

# Suppress some common torch/whisper warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class Ears:
    def __init__(self, model_size="tiny"):
        """
        Initializes the transcription module.
        model_size can be 'tiny', 'base', 'small', 'medium', or 'large'.
        'tiny' is fastest and best for initial testing.
        """
        print(f"Loading Whisper model '{model_size}' (this may take a moment on the first run)...")
        self.model = whisper.load_model(model_size)
        print("Model loaded successfully.")

    def transcribe(self, audio_path):
        """
        Transcribes the given audio file.
        """
        if not os.path.exists(audio_path):
            return f"Error: File {audio_path} not found."
            
        print(f"Transcribing {audio_path}...")
        try:
            result = self.model.transcribe(audio_path)
            return result["text"].strip()
        except Exception as e:
            return f"Transcription error: {str(e)}"

if __name__ == "__main__":
    # Create a dummy audio file (1 second of silence/sine wave) to test the pipeline
    import wave
    import struct
    import math

    test_audio = "test_audio.wav"
    print("Generating test audio file...")
    with wave.open(test_audio, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        # 1 second of a simple tone
        for i in range(16000):
            value = int(32767.0 * math.sin(2.0 * math.pi * 440.0 * (i / 16000.0)))
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)
    
    # Initialize Ears and transcribe
    ears = Ears(model_size="tiny")
    text = ears.transcribe(test_audio)
    
    print("\n--- Transcription Result ---")
    print(f"Text: '{text}'")
    print("----------------------------")
    
    # Clean up
    if os.path.exists(test_audio):
        os.remove(test_audio)
