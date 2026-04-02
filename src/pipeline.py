from src.retriever import Retriever
from src.generator import Generator

class RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline...")
        self.retriever = Retriever()
        self.generator = Generator()
        print("Pipeline ready!")
    
    def query(self, user_question, top_k=5):
        # step 1 - retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(
            user_question, top_k=top_k
        )
        
        # step 2 - generate answer
        answer = self.generator.generate(
            user_question, retrieved_chunks
        )
        
        return {
            "answer": answer,
            "sources": retrieved_chunks
        }
    
    def query_stream(self, user_question, top_k=5):
        # step 1 - retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(
            user_question, top_k=top_k
        )
        
        # step 2 - stream answer token by token
        # yield sources first so app.py can display them
        yield {"type": "sources", "data": retrieved_chunks}
        
        # then yield each token
        for token in self.generator.generate_stream(
            user_question, retrieved_chunks
        ):
            yield {"type": "token", "data": token}