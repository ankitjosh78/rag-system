# RAG System with DeepSeek 1.5B and Ollama

This project implements a **Retrieval-Augmented Generation (RAG)** system using the **DeepSeek 1.5B model** (via Ollama) for generating responses and **FAISS** for efficient document retrieval. The system supports querying over a collection of documents, including **text files**, **JSON files**, and other formats.

---

## Features

- **Document Retrieval**: Retrieve relevant documents using FAISS for fast similarity search.
- **Response Generation**: Generate responses using the DeepSeek 1.5B model via Ollama.
- **Document Support**: Supports text files (`.txt`), JSON files (`.json`), and other formats.
- **Modern GUI**: Built with `customtkinter` for a sleek and user-friendly interface.
- **Extensible**: Easily add support for new document formats or models.

---

## Prerequisites

Before running the project, ensure you have the following installed:

1. **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
2. **Ollama**: [Install Ollama](https://github.com/jmorganca/ollama)
3. **DeepSeek 1.5B Model**: Pull the model using Ollama:
   ```bash
   ollama run deepseek-r1:1.5b
   ```
  
## Project Structure
```
 rag-system/
├── documents/                  # Folder containing documents (JSON, TXT, etc.)
├── models.py                   # Code for interacting with Ollama API
├── retrieval.py                # FAISS-based document retrieval logic
├── rag.py                      # Core RAG system logic
├── gui.py                      # GUI code (built with customtkinter)
├── main.py                     # Entry point to run the application
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation 
```
## Installation
### Clone the repository:

```bash
git clone https://github.com/ankitjosh78/rag-system.git
cd rag-system
```
### Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
### Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
1. Start the Ollama Server
Ensure the Ollama server is running:

```bash
ollama serve
```
2. Add Documents
Place your documents (.txt, .json, etc.) in the documents folder. For example:

documents/document1.json:

```json
{
    "text": "Paris is the capital of France."
    "text": "Berlin is the capital of Germany."
}
```

documents/document2.txt:
```txt
Germany is known for its engineering and beer.
```

3. Run the Application
Start the RAG system GUI:

```bash
python main.py
```
4. Use the GUI
Enter a Query: Type your query in the input box and click "Submit".

View Results: The system will display the retrieved documents and generated response.

## Customization
### Add Support for New Document Formats
To add support for new file formats (e.g., .csv, .docx), create a new loading function in rag.py and update the load_documents_from_folder method.

Example for .csv:
```python
import pandas as pd

def load_csv(file_path):
    """Load and extract text from a CSV file."""
    df = pd.read_csv(file_path)
    return " ".join(df.iloc[:, 0])  # Assuming the first column contains text
```

### Change the Model
To use a different model with Ollama, update the model_name in models.py:

```python
self.model_name = "your-model-name"
```
