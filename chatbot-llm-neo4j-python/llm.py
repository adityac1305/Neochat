import streamlit as st

# Using Ollama LLM and Embeddings with Neo4j in Python
'''
from ollama import Ollama
from ollama import OllamaEmbeddings
from llm import llm

llm = Ollama(model="llama2")
embeddings = OllamaEmbeddings(model="sentence-transformer")
response = llm.predict ("")
print(response)
'''



# Using OpenAI LLM and Embeddings with Neo4j in Python


'''
# Create the LLM
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model=st.secrets["OPENAI_MODEL"],
)


# Create the Embedding model

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)



from llm import llm, embeddings
'''