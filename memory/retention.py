import os
import time

def prune_old_audio(directory="data/ambient", max_age_hours=48):
    """
    Deletes .wav files in the specified directory that are older than max_age_hours.
    This fulfills the Tier 1 Memory Lifecycle requirement (Hot Storage).
    """
    if not os.path.exists(directory):
        return
        
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            file_age = current_time - os.path.getmtime(filepath)
            
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                    print(f"[Retention] 🗑️ Deleted old audio file: {filename}")
                except Exception as e:
                    print(f"[Retention] ⚠️ Failed to delete {filename}: {e}")

if __name__ == "__main__":
    # Test script locally
    prune_old_audio()
