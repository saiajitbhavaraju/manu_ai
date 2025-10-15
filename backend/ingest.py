# backend/ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

load_dotenv()

# 1. Define paths and models
DATA_PATH = "data/" # Create a 'data' folder and put your PDFs there
QDRANT_PATH = "qdrant_db/"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # A good, fast, local model

def main():
    print("Starting data ingestion...")
    # 2. Load documents
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        print("No documents found. Please add your manuals to the 'data' folder.")
        return

    print(f"Loaded {len(documents)} documents.")

    # 3. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # 4. Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 5. Create and populate Qdrant database
    print("Creating Qdrant vector store...")
    qdrant = Qdrant.from_documents(
        texts,
        embeddings,
        path=QDRANT_PATH,
        collection_name="manuals",
    )
    print("Ingestion complete! Your Qdrant DB is ready.")

if __name__ == "__main__":
    main()