import customtkinter as ctk
from tkinter import scrolledtext
from rag import RAGSystem


class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG System with DeepSeek 1.5B and Ollama")
        self.root.geometry("800x600")  # Set window size

        # Configure customtkinter theme
        ctk.set_appearance_mode("dark")  # Options: "dark", "light", "system"
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

        # Create a frame for the input section
        self.input_frame = ctk.CTkFrame(root)
        self.input_frame.pack(pady=20, padx=20, fill="x")

        # Query label and entry
        self.query_label = ctk.CTkLabel(
            self.input_frame, text="Enter your query:", font=("Arial", 14)
        )
        self.query_label.pack(pady=10)

        self.query_entry = ctk.CTkEntry(self.input_frame, width=600, font=("Arial", 12))
        self.query_entry.pack(pady=10)

        # Submit button
        self.submit_button = ctk.CTkButton(
            self.input_frame,
            text="Submit",
            command=self.process_query,
            font=("Arial", 14),
        )
        self.submit_button.pack(pady=10)

        # Create a frame for the response section
        self.response_frame = ctk.CTkFrame(root)
        self.response_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Response area
        self.response_area = scrolledtext.ScrolledText(
            self.response_frame, width=80, height=20, font=("Arial", 12), wrap="word"
        )
        self.response_area.pack(pady=10, padx=10, fill="both", expand=True)

        # Initialize the RAG system
        self.rag_system = RAGSystem()

    def process_query(self):
        query = self.query_entry.get()
        response, retrieved_docs = self.rag_system.generate_response(query)

        # Clear the response area and display the results
        self.response_area.delete(1.0, ctk.END)
        self.response_area.insert(
            ctk.INSERT,
            f"Retrieved Documents:\n{retrieved_docs}\n\nResponse:\n{response}",
        )
