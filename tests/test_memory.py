import pytest
import os
from main import PersonalLLM
import time

def test_memory_storage_and_recall():
    app = PersonalLLM(debug=True)
    try:
        app.memory.client.delete_collection(app.memory.collection.name)
    except:
        pass
    app.memory.collection = app.memory.client.get_or_create_collection(app.memory.collection.name)
    
    # Store some dummy memory
    test_text = "[Speaker 1] 0.0-5.0: Hola, mi color favorito es el azul."
    app.memory.remember(test_text, metadata={"source": "test_audio.wav", "timestamp": "2026-04-06 12:00:00"})
    
    # Let chroma settle
    time.sleep(1)
    
    # Recall
    docs = app.memory.recall("color favorito", n_results=1)
    assert len(docs) > 0, "No memory recalled"
    assert "azul" in docs[0], f"Expected 'azul' in recalled memory, got {docs[0]}"

def test_rag_logic():
    app = PersonalLLM(debug=True)
    # Clear collection
    try:
        app.memory.client.delete_collection(app.memory.collection.name)
    except:
        pass
    app.memory.collection = app.memory.client.get_or_create_collection(app.memory.collection.name)
    
    test_text = "[2026-04-06 10:00:00]\n[Rubén] 0.0-5.0: Hoy voy a comer pizza."
    app.memory.remember(test_text, metadata={"source": "test.wav", "timestamp": "2026-04-06 10:00:00"})
    time.sleep(1)
    
    # Override brain's query to just test if context gets retrieved
    docs = app.memory.recall("¿Qué va a comer Rubén?", n_results=1)
    assert "pizza" in docs[0]
