# Patch for old SQLite versions
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import uuid
import os
import time

class Memory:
    def __init__(self, db_path="./local_memory"):
        """
        Initializes the local vector database.
        """
        # Create a persistent client that saves data to the local disk
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create a collection for our conversations
        self.collection = self.client.get_or_create_collection(name="conversations")

    def remember(self, text, metadata=None):
        """
        Stores a cleaned transcript in the memory database with a timestamp.
        """
        if not metadata:
            metadata = {}
            
        if "source" not in metadata:
            metadata["source"] = "unknown"
            
        # Automatically attach the current time as a UNIX timestamp
        if "timestamp" not in metadata:
            metadata["timestamp"] = int(time.time())
            
        doc_id = str(uuid.uuid4())
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"Memory saved! (ID: {doc_id})")
        return doc_id

    def recall(self, query, n_results=2):
        """
        Searches the memory for the most relevant past conversations.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Extract just the documents (the text) from the results
        if results and "documents" in results and len(results["documents"]) > 0:
            return results["documents"][0]
        return []

    def clear_all(self):
        """
        Completely wipes the memory database.
        """
        print("Wiping all memories...")
        self.client.delete_collection(name="conversations")
        self.collection = self.client.get_or_create_collection(name="conversations")
        print("Brain is now completely wiped clean.")

    def expire_older_than(self, days):
        """
        Deletes any memories older than the specified number of days.
        """
        cutoff_time = int(time.time()) - (days * 24 * 60 * 60)
        
        # Fetch the IDs of memories that are older than our cutoff time
        try:
            results = self.collection.get(
                where={"timestamp": {"$lt": cutoff_time}}
            )
            
            if results and results["ids"]:
                old_ids = results["ids"]
                self.collection.delete(ids=old_ids)
                print(f"Successfully deleted {len(old_ids)} expired memories (older than {days} days).")
            else:
                print(f"No memories found older than {days} days.")
        except Exception as e:
            print("Could not process expiration. Note: Memories saved before the timestamp update cannot be expired by time.")

if __name__ == "__main__":
    mem = Memory()
    
    # Let's save some fake past conversations
    print("Saving memories...")
    mem.remember("On Monday, I discussed setting up an offline LLM using Ollama and ChromaDB.", {"date": "Monday"})
    mem.remember("I need to buy milk, eggs, and a new GPU for my computer.", {"date": "Tuesday"})
    mem.remember("The car needs an oil change by next week. The mechanic's name is Bob.", {"date": "Wednesday"})
    
    # Now let's ask a question to retrieve the relevant memory
    query = "Who is going to fix my vehicle?"
    print(f"\nQuerying memory for: '{query}'")
    
    results = mem.recall(query, n_results=1)
    
    print("\n--- RECALLED MEMORY ---")
    for r in results:
        print(f"- {r}")
    print("-----------------------")
