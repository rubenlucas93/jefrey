import httpx
import json

class Brain:
    def __init__(self, base_url="http://localhost:11434", model="llama3.2"):
        self.base_url = base_url
        self.model = model

    def query(self, prompt, system_prompt="You are a helpful assistant."):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False
        }
        
        try:
            response = httpx.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            return response.json().get("response", "No response from brain.")
        except Exception as e:
            return f"Error connecting to Brain: {str(e)}"

if __name__ == "__main__":
    # Test the brain
    brain = Brain()
    print(f"Testing Brain with model: {brain.model}")
    print(brain.query("Say hello!"))
