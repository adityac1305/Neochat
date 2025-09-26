from llm import llm, embeddings

# Testing def predict 
print(llm.predict("Tell me a joke."))


# Testing def generate
print(llm.generate("Tell me a joke."))


# Testing def chat
messages = [
    {"role": "system", "content": "You are a helpful math assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."},
    {"role": "user", "content": "Now multiply it by 10."}
]

resp = llm.chat(messages)
print(resp)



# Testing def embed_text
vec = embeddings.embed_text("Graph databases rock!")
print(f"Embedding length = {len(vec)}")
print(vec[:5]) 


# Testing def embed_documents
docs = [
    "Neo4j is a graph database.",
    "PostgreSQL is a relational database.",
    "Ollama runs LLMs locally."
]

vectors = embeddings.embed_documents(docs)

print(f"Number of embeddings: {len(vectors)}")
print(f"Length of one embedding: {len(vectors[0])}")
print(f"First vector sample: {vectors[0][:5]}")