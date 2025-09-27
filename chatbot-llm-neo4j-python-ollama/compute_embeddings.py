from graph import graph
import ollama
import streamlit as st

# The model will generate embeddings with 384 dimensions
model = st.secrets["OLLAMA_EMBEDDING_MODEL"]


def compute_embedding(text: str):
    # Generate a 384-dim embedding for given text
    result = ollama.embeddings(model=model, prompt=text)
    return result["embedding"]

def update_movie_embeddings():
    # Fetch all movies and update their set and assign the plotEmbedding384 property
    
    movies = graph.query("MATCH (m:Movie) WHERE m.plot IS NOT NULL RETURN m.title AS title, m.plot AS plot")
    
    for movie in movies:
        title = movie["title"]
        plot = movie["plot"]
        
        embedding = compute_embedding(plot)
        
        graph.query(
            """
            MATCH (m:Movie {title: $title})
            SET m.plotEmbedding384 = $embedding
            """,
            {"title": title, "embedding": embedding},
        )
        
        print(f" Updated embedding for movie: {title} (dim={len(embedding)})")

if __name__ == "__main__":
    update_movie_embeddings()