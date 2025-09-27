import streamlit as st
import ollama
from langchain_core.runnables import RunnableLambda


OLLAMA_MODEL = st.secrets["OLLAMA_MODEL"]
OLLAMA_EMBEDDING_MODEL = st.secrets["OLLAMA_EMBEDDING_MODEL"]
OLLAMA_HOST = st.secrets.get("OLLAMA_HOST", None)


def _extract_text(resp):
    if resp is None:
        return None
    if hasattr(resp, "response"):
        return getattr(resp, "response")
    if hasattr(resp, "text"):
        return getattr(resp, "text")
    if isinstance(resp, dict):
        return resp.get("response") or resp.get("text") or str(resp)
    return str(resp)


def _extract_embedding(resp):
    if resp is None:
        return None
    emb = getattr(resp, "embedding", None)
    if emb is None and isinstance(resp, dict):
        emb = resp.get("embedding") or resp.get("embeddings")
    if emb is None:
        return None
    return list(emb)


def clean_for_react(output: str) -> str:
    text = output.strip()
    if text.lower().startswith("assistant:"):
        text = text.split(":", 1)[1].strip()
    if not any(x in text for x in ["Thought:", "Action:", "Final Answer:"]):
        return f"Final Answer: {text}"
    return text




class OllamaLLM:
    def __init__(self, model: str, host: str | None = None):
        self.model = model
        self.host = host

    ### Use predict when you only care about the answer text (simpler for chatbot UI). ###

    '''
    Example usage:

    resp = llm.predict("What is 2+2?")
    print(resp)

    # As we have the extract_text function, we just get the text response:
    The answer is 4. 
    
    '''


    def predict(self, prompt: str, **kwargs):
        if hasattr(prompt, "to_string"):
            prompt = prompt.to_string()

        allowed_kwargs = {k: v for k, v in kwargs.items() if k in ["temperature", "top_p", "n"]}
        if hasattr(ollama, "generate"):
            out = ollama.generate(self.model, prompt, **allowed_kwargs)
            text = _extract_text(out)
            return clean_for_react(text)

        client = ollama.Client(host=self.host) if self.host else ollama.Client()
        out = client.generate(self.model, prompt, **allowed_kwargs)
        text = _extract_text(out)
        return clean_for_react(text)

   

    ### Use generate when you want extra metadata (e.g., for logging, debugging, token usage, or structured responses). ###

    '''
    Example usage:
    
    resp = llm.generate("What is 2+2?")
    print(resp)

    # Full response object with metadata:

    {
    "model": "llama2",
    "created_at": "2025-09-24T08:22:00Z",
    "response": "The answer is 4."
    }

    '''

    def generate(self, prompt: str, **kwargs):
        if hasattr(ollama, "generate"):
            return ollama.generate(self.model, prompt, **kwargs)
        client = ollama.Client(host=self.host) if self.host else ollama.Client()
        return client.generate(self.model, prompt, **kwargs)
    
   
    ### Use chat when you want to have a multi-turn conversation with context. ###
    
    '''
    Example usage:

    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "Now multiply it by 10."}
    ]

    resp = llm.chat(messages)

    print(resp)

    # Full response object with metadata:

    {
        "model": "llama2",
        "created_at": "2025-09-24T08:45:00Z",
        "message": {
            "role": "assistant",
            "content": "That would be 40."
        }
    }
    
    '''

    def chat(self, messages, **kwargs):
        if hasattr(ollama, "chat"):
            return ollama.chat(self.model, messages, **kwargs)
        client = ollama.Client(host=self.host) if self.host else ollama.Client()
        return client.chat(self.model, messages, **kwargs)


class OllamaEmbeddings:
    def __init__(self, model: str, host: str | None = None):
        self.model = model
        self.host = host

    def embed_text(self, text: str, **kwargs):
        if hasattr(ollama, "embeddings"):
            out = ollama.embeddings(self.model, text, **kwargs)
            return _extract_embedding(out)
        if hasattr(ollama, "embed"):
            out = ollama.embed(self.model, text, **kwargs)
            return _extract_embedding(out)
        client = ollama.Client(host=self.host) if self.host else ollama.Client()
        if hasattr(client, "embeddings"):
            out = client.embeddings(self.model, text, **kwargs)
            return _extract_embedding(out)
        if hasattr(client, "embed"):
            out = client.embed(self.model, text, **kwargs)
            return _extract_embedding(out)
        return None

    def embed_documents(self, documents: list[str], **kwargs):
        return [self.embed_text(d, **kwargs) for d in documents]


llm = OllamaLLM(OLLAMA_MODEL, host=OLLAMA_HOST)
embeddings = OllamaEmbeddings(OLLAMA_EMBEDDING_MODEL, host=OLLAMA_HOST)
llm_runnable = RunnableLambda(lambda prompt, **kwargs: llm.predict(prompt, **kwargs))

__all__ = ["llm_runnable", "embeddings", "llm"]























'''

import streamlit as st

### Using Ollama LLM and Embeddings with Neo4j in Python ###

# Configure via .streamlit/secrets.toml
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL")
OLLAMA_EMBEDDING_MODEL = st.secrets.get("OLLAMA_EMBEDDING_MODEL")
OLLAMA_HOST = st.secrets.get("OLLAMA_HOST", None)  # e.g. "http://localhost:11434"

try:
    from ollama import Ollama, OllamaEmbeddings
except Exception as e:
    raise ImportError(
        "The 'ollama' package is required. Install it into your venv with 'pip install ollama'."
    ) from e

client_kwargs = {}
if OLLAMA_HOST:
    client_kwargs["base_url"] = OLLAMA_HOST

# Create objects named `llm` and `embeddings` for the rest of the app to import
llm = Ollama(model=OLLAMA_MODEL, **client_kwargs)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, **client_kwargs)

__all__ = ["llm", "embeddings"]



### Using OpenAI LLM and Embeddings with Neo4j in Python ###



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