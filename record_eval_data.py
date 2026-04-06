import os
import glob
import json
from ears.recorder import record_audio

def main():
    eval_dir = "data/eval"
    if not os.path.exists(eval_dir):
        print(f"Error: Directory '{eval_dir}' not found.")
        return

    # Find all _truth.json templates
    json_files = sorted(glob.glob(os.path.join(eval_dir, "*_truth.json")))
    
    if not json_files:
        print(f"No *_truth.json files found in {eval_dir}.")
        return

    print("🎙️  Evaluation Audio Recording Tool  🎙️")
    print("=========================================")

    for json_file in json_files:
        base_name = os.path.basename(json_file).replace("_truth.json", "")
        audio_file = os.path.join(eval_dir, f"{base_name}.wav")
        
        if os.path.exists(audio_file):
            print(f"\n[SKIP] Audio already exists for: {base_name}.wav")
            continue

        with open(json_file, "r") as f:
            data = json.load(f)
            
        print("\n" + "="*50)
        print(f"📄 TEMPLATE: {base_name}")
        print("Please read the following conversation out loud (with your partner if possible):\n")
        
        for line in data.get("transcript", []):
            print(f"  {line}")
        
        print("\nPress ENTER when you are ready to record. Press Ctrl+C when you are done speaking.")
        input()
        
        try:
            record_audio(audio_file)
            print(f"✅ Saved to: {audio_file}")
        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
            break
        except Exception as e:
            print(f"Error recording: {e}")

    print("\nAll missing audio files have been recorded! You can now run the eval loop.")

if __name__ == "__main__":
    main()
