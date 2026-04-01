import sounddevice as sd
import soundfile as sf
import queue
import sys

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
