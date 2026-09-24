import os
import glob
from dotenv import load_dotenv
load_dotenv(".env")

from fastembed import TextEmbedding
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import time

print("Initializing Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "roadlaw-legal"
index = pc.Index(index_name)

print("Clearing old vectors...")
# To delete all vectors, we can delete the index and recreate it, or delete by namespace if we have one.
# Since we didn't use a specific namespace, we might need to delete all vectors, but the Pinecone free tier allows only 1 index.
# The safest way is to fetch all IDs or just delete all if supported.
try:
    index.delete(delete_all=True)
except Exception as e:
    print("Warning on delete:", e)
    pass
time.sleep(2)

print("Loading FastEmbed model (BAAI/bge-base-en-v1.5)...")
model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")

pdf_files = glob.glob("legal_pdfs/*.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

total_chunks = 0
for pdf in pdf_files:
    print(f"Processing: {pdf}")
    loader = PyPDFLoader(pdf)
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    
    # Process in batches
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c.page_content for c in batch]
        metadatas = []
        for j, c in enumerate(batch):
            meta = c.metadata.copy()
            meta["text"] = c.page_content
            meta["country"] = "India"
            meta["level"] = "national"
            meta["state_name"] = "ALL"
            meta["source"] = os.path.basename(pdf)
            metadatas.append(meta)
            
        embeddings = [e.tolist() for e in model.embed(texts)]
        ids = [f"{os.path.basename(pdf)}_chunk_{i+j}" for j in range(len(batch))]
        
        vectors = list(zip(ids, embeddings, metadatas))
        index.upsert(vectors=vectors)
        total_chunks += len(batch)
        print(f"  Upserted {total_chunks} total chunks...")

print(f"\nDONE! Upserted {total_chunks} chunks.")
