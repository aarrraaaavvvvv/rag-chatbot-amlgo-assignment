from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

class Retriever:
    def __init__(self):
        print("Loading retriever...")
        
        # load the embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # load the FAISS index
        self.index = faiss.read_index("vectordb/faiss_index.index")
        
        # load the chunks
        with open("chunks/chunks.json", "r") as f:
            self.chunks = json.load(f)
        
        print(f"Retriever ready. {self.index.ntotal} vectors loaded.")
    
    def retrieve(self, query, top_k=3):
        # convert the user question into an embedding
        query_embedding = self.model.encode([query])
        
        # search FAISS for top_k most similar chunks
        distances, indices = self.index.search(
            np.array(query_embedding), top_k
        )
        
        # return the actual text chunks
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "chunk": self.chunks[idx],
                "distance": float(distances[0][i]),
                "chunk_id": int(idx)
            })
        
        return results


# TEST - run this file directly to test retriever
if __name__ == "__main__":
    retriever = Retriever()
    
    test_query = "What happens if eBay suspends my account?"
    print(f"\nQuery: {test_query}")
    print("\nTop 3 relevant chunks:")
    
    results = retriever.retrieve(test_query)
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (distance: {r['distance']:.4f}) ---")
        print(r['chunk'][:300])