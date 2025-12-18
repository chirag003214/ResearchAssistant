from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
import os
from dotenv import load_dotenv

load_dotenv()

def init_settings():
    """
    Initializes LlamaIndex settings with specific Embedding models and LLMs.
    """
    # 1. Embeddings: Still using local HuggingFace (Free & Fast)
    print("⚙️  Loading Embedding Model (Local)...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 2. LLM: Switched to the newest Groq model
    print("⚙️  Loading LLM (Groq Llama 3.3)...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY is missing from .env file!")

    # UPDATED MODEL NAME HERE:
    llm = Groq(model="llama-3.3-70b-versatile", api_key=api_key)

    # 3. Apply to Global Settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    return Settings