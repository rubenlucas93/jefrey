import argparse
import queue
import threading
import os
import requests
import sys

from ears.recorder import ambient_recorder_thread

def upload_worker(audio_queue, server_url):
    """Takes audio chunks from the queue and uploads them to the server."""
    while True:
        filename = audio_queue.get()
        try:
            print(f"  [Uploading] Sending {filename} to Server...")
            with open(filename, 'rb') as f:
                files = {'file': (os.path.basename(filename), f, 'audio/wav')}
                response = requests.post(f"{server_url}/ingest", files=files)
            
            if response.status_code == 200:
                print(f"  [Success] Server received and processed {filename}")
                os.remove(filename)  # Delete locally to save space on client
            else:
                print(f"  [Error] Server rejected file: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"  [Connection Error] Could not reach server: {e}")
        finally:
            audio_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description="Personal LLM - Living Room Client")
    parser.add_argument("--host", type=str, default="http://127.0.0.1:8000", help="Server URL (e.g. http://192.168.1.100:8000)")
    args = parser.parse_args()

    server_url = args.host.rstrip('/')
    
    # 1. Test Server Connection
    print(f"Testing connection to server at {server_url}...")
    try:
        requests.get(server_url, timeout=3)
        # Note: FastAPI without a base GET route returns 404, but it proves the server is up!
    except requests.exceptions.RequestException:
        print(f"⚠️ Warning: Could not connect to {server_url}. Ensure server is running and IP is correct.")

    audio_queue = queue.Queue()
    
    # 2. Start the ambient background recorder
    rec_thread = threading.Thread(target=ambient_recorder_thread, args=(audio_queue, 30, "client_ambient", 0.002), daemon=True)
    rec_thread.start()
    
    # 3. Start the uploader thread
    up_thread = threading.Thread(target=upload_worker, args=(audio_queue, server_url), daemon=True)
    up_thread.start()

    print("\n" + "="*50)
    print("💻 CLIENT MODE ACTIVE (LIVING ROOM)")
    print(f"Connected to Server: {server_url}")
    print("Microphone is open. Audio is chunked and sent automatically.")
    print("Type a question below to query the server's Brain.")
    print("="*50 + "\n")

    # 4. Main prompt loop for querying
    while True:
        try:
            question = input("Ask the Brain (or 'quit'): ")
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if question.strip():
                try:
                    res = requests.post(f"{server_url}/ask", json={"text": question})
                    if res.status_code == 200:
                        print(f"\n🤖 Server Answer:\n{res.json().get('answer')}\n")
                    else:
                        print(f"❌ Server Error: {res.status_code}")
                except Exception as e:
                    print(f"❌ Connection error: {e}")
        except KeyboardInterrupt:
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
