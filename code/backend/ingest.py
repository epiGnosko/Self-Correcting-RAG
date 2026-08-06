from llm import embeddings, llm
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# ---------------------------------------------------
# Load all PDFs
# ---------------------------------------------------

loader = PyPDFLoader("2401.15884v3.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages.")

# ---------------------------------------------------
# Split into chunks
# ---------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# ---------------------------------------------------
# Create Chroma Vector Store
# ---------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector database created successfully!")
print(f"Stored {len(chunks)} chunks.")