import pytest
import os
from ears.transcriber import Ears

def test_transcriber_basic():
    """
    Tests if Whisper can load the model and run transcription on an existing file.
    Uses 'base' for faster testing to avoid waiting for 'small' downloads in CI.
    """
    ears = Ears(model_size="tiny")
    
    # We expect user_recording.wav to exist from previous manual tests
    audio_file = "user_recording.wav"
    if not os.path.exists(audio_file):
        pytest.skip(f"{audio_file} not found, skipping transcription test")
        
    result = ears.transcribe(audio_file, language="es")
    
    assert "text" in result
    assert "segments" in result
    assert isinstance(result["segments"], list)
    assert len(result["text"]) > 0, "Transcription should not be empty"
