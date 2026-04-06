import sounddevice as sd
import soundfile as sf
import queue
import sys
import numpy as np
import os
from datetime import datetime

def ambient_recorder_thread(audio_queue, chunk_duration=30, output_dir="data/ambient", silence_threshold=0.002):
    """
    Runs in the background, slicing audio into continuous chunks.
    Pushes filename to audio_queue if the chunk is not completely silent.
    """
    os.makedirs(output_dir, exist_ok=True)
    samplerate = 16000
    channels = 1
    
    q = queue.Queue()
    def callback(indata, frames, time, status):
        if status:
            pass # ignore underflows in background
        q.put(indata.copy())

    print(f"\n[Ambient] 🎙️ Microphone open. Listening in background...")
    
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        while True:
            frames_needed = samplerate * chunk_duration
            frames_collected = 0
            audio_data = []
            
            # Wait and collect chunk_duration seconds of audio
            while frames_collected < frames_needed:
                data = q.get()
                audio_data.append(data)
                frames_collected += len(data)
                
            audio_concat = np.concatenate(audio_data)
            rms = np.sqrt(np.mean(audio_concat**2))
            
            # Only save and process if it surpasses the silence threshold
            if rms > silence_threshold:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"ambient_{timestamp}.wav")
                sf.write(filename, audio_concat, samplerate, subtype='PCM_16')
                audio_queue.put(filename)

def record_audio(filename="user_recording.wav", samplerate=16000, channels=1):
    """
    Records audio from the default microphone until the user presses Ctrl+C.
    """
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called for each audio block by sounddevice."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("\n" + "="*50)
    print("🎙️  RECORDING NOW...")
    print("Speak into your microphone.")
    print("Press Ctrl+C when you are finished.")
    print("="*50 + "\n")

    try:
        # We use mode='w' to overwrite the file if it exists
        with sf.SoundFile(filename, mode='w', samplerate=samplerate,
                          channels=channels, subtype='PCM_16') as file:
            with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
                while True:
                    file.write(q.get())
    except KeyboardInterrupt:
        print("\n⏹️  Recording stopped and saved to", filename)
        return filename
    except Exception as e:
        print(f"\n❌ Error accessing microphone: {e}")
        return None

if __name__ == "__main__":
    # Test the recorder by running it directly
    record_audio()
