import streamlit as st


from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph(
    url=st.secrets["NEO4J_URI"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"],
)

'''
We can import this in any part/file of project using the following  
from graph import graph

'''