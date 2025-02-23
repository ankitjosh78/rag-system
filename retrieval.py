import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def build_faiss_index(document_embeddings):
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(document_embeddings)
    return index


def retrieve_documents(query, index, documents, top_k=3):
    # Encode the query into an embedding
    query_embedding = embedding_model.encode([query])

    # Search the FAISS index for the most similar documents
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]
