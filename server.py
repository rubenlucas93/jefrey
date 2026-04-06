import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from main import PersonalLLM
from memory.retention import prune_old_audio

app = FastAPI()

# Initialize the Personal LLM System once on startup
print("Initializing PersonalLLM Server...")
llm = PersonalLLM(debug=False)
llm.skip_mapping = True  # We want it to use biometrics, not ask for input

os.makedirs("data/server_ambient", exist_ok=True)

class Question(BaseModel):
    text: str

@app.post("/ingest")
async def ingest_audio(file: UploadFile = File(...)):
    """Receives a .wav file from the client and processes it."""
    file_location = f"data/server_ambient/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Auto-delete audio older than 48 hours to save space
    prune_old_audio(directory="data/server_ambient", max_age_hours=48)
    
    # Send it through the LLM pipeline
    try:
        print(f"[Server] Processing uploaded file: {file.filename}")
        llm.ingest_audio(file_location)
    except Exception as e:
        print(f"[Server] Error processing audio: {e}")
        return {"status": "error", "message": str(e)}

    return {"status": "success", "filename": file.filename}

@app.post("/ask")
async def ask_question(q: Question):
    """Receives a question text, queries the Brain, and returns the answer."""
    print(f"\n[Server] Received question from client: {q.text}")
    try:
        answer = llm.ask(q.text)
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"Error thinking: {str(e)}"}

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 SERVER MODE ACTIVE")
    print("Listening for audio chunks and questions on port 8000...")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
