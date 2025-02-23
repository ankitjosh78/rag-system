import requests


class OllamaAPI:
    def __init__(self, model_name="deepseek-r1:1.5b"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api/generate"  # Ollama's API endpoint

    def generate(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,  # Set to True if you want streaming responses
        }
        response = requests.post(self.base_url, json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
