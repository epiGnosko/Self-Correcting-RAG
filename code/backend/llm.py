from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5", api_key=api_key
)