import streamlit as st
from llm import llm_runnable, embeddings
from graph import graph

# Create the Neo4jVector


from langchain_neo4j import Neo4jVector

neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              # (1)
    graph=graph,                             # (2)
    index_name="moviePlots384",                 # (3)
    node_label="Movie",                      # (4)
    text_node_property="plot",               # (5)
    embedding_node_property="plotEmbedding384", # (6)
    retrieval_query="""
RETURN
    node.plot AS text,
    score,
    {
        title: node.title,
        directors: [ (person)-[:DIRECTED]->(node) | person.name ],
        actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
        tmdbId: node.tmdbId,
        source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
    } AS metadata
"""
)




# Create the retriever

retriever = neo4jvector.as_retriever()


# Create the prompt

from langchain_core.prompts import ChatPromptTemplate

instructions = (
    "Use the given context to answer the question.\n"
    "Always answer in plain text, no Thought/Action/Observation format.\n"
    "If links are available in metadata, return them as clickable URLs.\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)


# Create the chain 

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

question_answer_chain = create_stuff_documents_chain(llm_runnable, prompt)
plot_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)


# Create a function to call the chain

def get_movie_plot(input):
    return plot_retriever.invoke({"input": input})