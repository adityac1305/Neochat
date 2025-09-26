import ollama
model = "koill/sentence-transformers:all-minilm-l12-v2"
print("Models local:", ollama.list())
print("Embedding test:", ollama.embeddings(model, "this is a test"))