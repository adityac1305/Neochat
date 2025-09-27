import streamlit as st
import ollama
model = st.secrets["OLLAMA_EMBEDDING_MODEL"]
print("Models local:", ollama.list())

result = ollama.embeddings(model, ["this is a test"])
embedding = result["embedding"]
dimensions = len(embedding)
print(f"Embedding dimensions: {dimensions}")

print("Embedding test:", ollama.embeddings(model, "this is a test"))