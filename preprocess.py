from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# STEP 1 - Extract text from PDF
print("Extracting text from PDF...")
reader = PdfReader("data/training_document.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"
print(f"Total characters extracted: {len(full_text)}")

# STEP 2 - Chunk the text
print("Chunking text...")
def chunk_text(text, chunk_size=1000, chunk_overlap=90):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period != -1:
                end = start + last_period + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - chunk_overlap
    return chunks

chunks = chunk_text(full_text)
print(f"Total chunks created: {len(chunks)}")

# STEP 3 - Save chunks to file
print("Saving chunks...")
with open("chunks/chunks.json", "w") as f:
    json.dump(chunks, f)
print("Chunks saved to chunks/chunks.json")

# STEP 4 - Generate embeddings
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Generating embeddings... this may take a minute")
embeddings = model.encode(chunks, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")

# STEP 5 - Save to FAISS
print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
faiss.write_index(index, "vectordb/faiss_index.index")
print(f"FAISS index saved. Total vectors: {index.ntotal}")

print("\nAll done! Your vector database is ready.")