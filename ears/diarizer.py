import os
import torch
import torchaudio

# Monkeypatch torchaudio for compatibility with older pyannote versions
if not hasattr(torchaudio, "set_audio_backend"):
    # This function was removed in torchaudio 2.1.0 but older pyannote calls it
    torchaudio.set_audio_backend = lambda backend: None

class Diarizer:
    def __init__(self, hf_token="", language="en"):
        """
        Uses pyannote.audio to identify speakers based on actual audio frequencies.
        """
        self.language = language
        self.hf_token = hf_token
        self.pipeline = None

        if self.hf_token:
            print("Loading Pyannote Diarization Pipeline (this requires an internet connection for the first run)...")
            try:
                # Monkeypatch huggingface_hub to handle old pyannote calling with use_auth_token
                import huggingface_hub
                original_download = huggingface_hub.hf_hub_download
                
                def patched_download(*args, **kwargs):
                    if "use_auth_token" in kwargs:
                        kwargs["token"] = kwargs.pop("use_auth_token")
                    return original_download(*args, **kwargs)
                
                huggingface_hub.hf_hub_download = patched_download
                
                # Monkeypatch np.NaN for older pyannote versions running on Numpy 2.0+
                import numpy as np
                if not hasattr(np, "NaN"):
                    np.NaN = np.nan

                from pyannote.audio import Pipeline
                from pyannote.audio.core import io as pyannote_io

                # Monkeypatch AudioDecoder if it is missing in the installed pyannote version
                if not hasattr(pyannote_io, "AudioDecoder"):
                    # We create a dummy class to prevent the NameError
                    class DummyAudioDecoder:
                        def __init__(self, *args, **kwargs):
                            self.metadata = None
                    pyannote_io.AudioDecoder = DummyAudioDecoder

                # Set the environment variable for Hugging Face to automatically pick up the token
                os.environ["HF_TOKEN"] = self.hf_token

                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                )

                # Force CPU usage due to outdated NVIDIA drivers
                self.pipeline.to(torch.device("cpu"))

                print("Pyannote loaded successfully.")
            except ImportError:
                print("Error: pyannote.audio is not installed. Please install it via requirements.")
            except Exception as e:
                print(f"Error loading pyannote pipeline. Did you accept the HF conditions and provide a valid token? Details: {e}")
        else:
            print("Warning: No hf_token provided in config.json. Diarization will be disabled.")

    def tag_speakers(self, audio_path, whisper_segments):
        """
        Matches Whisper's text segments with Pyannote's speaker timestamps.
        """
        if not whisper_segments:
            return ""
            
        if not self.pipeline:
            print("Skipping diarization (Pipeline not initialized).")
            # Fallback: Just return the segments without speaker tags
            fallback = []
            for seg in whisper_segments:
                fallback.append(f"[Speaker ?] {seg['start']:.1f}-{seg['end']:.1f}: {seg['text'].strip()}")
            return "\n".join(fallback)

        print("Analyzing audio frequencies for speaker separation...")
        # Run pyannote on the audio file (force exactly 1 or 2 speakers if possible but we don't know).
        # We let pyannote decide automatically.
        diarization = self.pipeline(audio_path)

        tagged_transcript = []
        for seg in whisper_segments:
            seg_start = seg['start']
            seg_end = seg['end']
            text = seg['text'].strip()
            
            # Find the speaker who spoke the most during this Whisper segment
            best_speaker = "UNKNOWN"
            max_overlap = 0.0
            
            # itertracks yields: (Segment(start, end), track_id, speaker_label)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Calculate time overlap
                overlap_start = max(seg_start, turn.start)
                overlap_end = min(seg_end, turn.end)
                overlap = max(0.0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker
                    
            # Simplify Pyannote's "SPEAKER_00" to "Speaker 1"
            if best_speaker.startswith("SPEAKER_"):
                try:
                    num = int(best_speaker.split("_")[1]) + 1
                    best_speaker = f"Speaker {num}"
                except:
                    pass
            
            # Fallback if pyannote found no overlap
            if best_speaker == "UNKNOWN":
                 best_speaker = "Speaker 1"

            tagged_transcript.append(f"[{best_speaker}] {seg_start:.1f}-{seg_end:.1f}: {text}")
            
        return "\n".join(tagged_transcript)
