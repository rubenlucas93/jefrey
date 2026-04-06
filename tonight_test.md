# Tonight's Test: Living Room Client-Server Ambient Mode

You have a powerful Desktop computer (The Brain) and a lightweight Laptop (The Ear) in your living room. We have officially split the architecture so you can test this smoothly.

## Step 1: Start the Server (On your Big Desktop Computer)
This machine will handle the heavy lifting: Whisper (Medium), Pyannote Diarization, ChromaDB, and Ollama.

1.  Find the Local IP address of your Desktop (usually something like `192.168.1.XX`).
    *   *Linux/Mac:* Run `ip addr` or `ifconfig`.
    *   *Windows:* Run `ipconfig`.
2.  Activate the virtual environment: `source venv/bin/activate`
3.  Run the server: 
    ```bash
    python server.py
    ```
4.  Wait until you see the message: `🚀 SERVER MODE ACTIVE. Listening for audio chunks and questions on port 8000...`

## Step 2: Setup the Client (On your Living Room Laptop)
This machine will only use the microphone to record audio chunks and send them over Wi-Fi.

1.  Make sure you copy the `client.py`, `ears/recorder.py` and `requirements.txt` to the laptop (or just clone the repo).
2.  Create a virtual environment and install only the lightweight dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install sounddevice soundfile numpy requests
    ```
3.  Run the client and point it to the Server's IP address:
    ```bash
    python client.py --host http://<YOUR_DESKTOP_IP>:8000
    ```
    *(Example: `python client.py --host http://192.168.1.45:8000`)*

## Step 3: Run the Test
Once both are connected, just sit in your living room and talk normally.
*   **The Recorder:** Every 30 seconds of active speech, the laptop will create a `.wav` file, upload it to your desktop, and immediately delete its local copy to save space.
*   **The Processor:** Your desktop will receive the file, transcribe it with the new `Whisper Medium` model, identify you using the new `0.55` threshold, and store it in ChromaDB. Old `.wav` files on the server will be automatically deleted if they exceed 48 hours.
*   **The Questions:** You can type questions directly into the laptop's terminal at any time (e.g., *"¿Qué acabo de decir sobre el trabajo?"*). The laptop will ping the desktop, do the RAG lookup, and print the answer.

## What You Need to Verify Tomorrow:
1.  **Transcription Accuracy:** Did `Whisper Medium` fix errors like *napolas/amapolas*?
2.  **Voice ID:** Did the new `0.55` threshold recognize you without needing to type your name, even when your tone changed or you were further from the mic?
3.  **Storage:** Look inside `data/server_ambient/` on your desktop. Verify that the files are being ingested and space isn't blowing up unexpectedly.
4.  **Connectivity:** Did the client/server HTTP communication feel fast enough for your Q&A?

Check these off and let me know the results so we can move on to Proactivity (Point 2)!
