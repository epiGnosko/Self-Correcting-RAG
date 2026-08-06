from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    temperature=0.2,
    api_key=api_key
)

embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5", api_key=api_key
)