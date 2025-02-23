from models import OllamaAPI
from retrieval import build_faiss_index, retrieve_documents
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os


class RAGSystem:
    def __init__(self):
        self.ollama = OllamaAPI(model_name="deepseek-r1:1.5b")

        # Load your documents
        self.documents = self.load_documents_from_folder(
            "documents"
        )  # Folder containing JSON files

        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.document_embeddings = embedding_model.encode(self.documents)

        # Build the FAISS index
        self.index = build_faiss_index(self.document_embeddings)

    def load_documents_from_folder(self, folder_path):
        """Load all documents (JSON, TXT) from a folder."""
        documents = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".json"):
                text = self.load_json(file_path)
            elif file_name.endswith(".txt"):
                with open(file_path, "r") as file:
                    text = file.read()
            else:
                continue  # Skip unsupported file types
            documents.append(text)
        return documents

    def load_json(self, file_path):
        """Load and extract text from a JSON file."""
        with open(file_path, "r") as file:
            data = json.load(file)
            # Assuming the JSON structure contains a "text" or "content" field
            if isinstance(data, dict):
                return data.get("text", "") or data.get("content", "")
            elif isinstance(data, list):
                return " ".join(
                    [item.get("text", "") or item.get("content", "") for item in data]
                )
            else:
                return str(data)  # Fallback for other JSON structures

    def generate_response(self, query):
        # Retrieve relevant documents
        retrieved_docs = retrieve_documents(query, self.index, self.documents)

        # Combine query and retrieved documents into a prompt
        context = " ".join(retrieved_docs)
        input_text = f"Query: {query}\nContext: {context}"

        # Generate response using Ollama
        response = self.ollama.generate(input_text)
        return response, retrieved_docs
