from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from dotenv import load_dotenv
load_dotenv()
# Use a fast 7B/1B model from your active list and set timeout
model_name = "meta/llama-3.2-1b-instruct"
print(f"Testing fast model: {model_name}...")

llm = ChatNVIDIA(
    model=model_name,
    max_tokens=50,
    api_key=os.getenv("NVIDIA_API_KEY")
)
try:
    print("Response stream: ", end="", flush=True)
    # Using stream instead of invoke
    for chunk in llm.stream("Respond with the exact words: 'API Connected Successfully'"):
        print(chunk.content, end="", flush=True)
    print("\n\n--- SUCCESS ---")
except Exception as e:
    print("\n\n--- ERROR ---")
    print(e)