from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
    
    def build_prompt(self, query, retrieved_chunks):
        # combine retrieved chunks into context
        context = "\n\n".join([
            f"[Source {i+1}]:\n{chunk['chunk']}" 
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        prompt = f"""You are a friendly and helpful assistant specializing in eBay's User Agreement.

You have two modes:
1. If the user is making small talk or greeting you, respond in a warm, friendly and conversational way. Keep it brief and invite them to ask about the eBay User Agreement.
2. If the user is asking a question related to eBay, answer based ONLY on the provided context below. If the answer is not in the context, say "I couldn't find specific information about that in the eBay User Agreement."

Always be warm, clear and concise. When the context contains the answer, be direct and specific. Do not hedge or redirect unnecessarily.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:"""
        
        return prompt
    
    def generate(self, query, retrieved_chunks):
        prompt = self.build_prompt(query, retrieved_chunks)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.1  # low temperature = more factual, less creative
        )
        
        return response.choices[0].message.content
    
    def generate_stream(self, query, retrieved_chunks):
        prompt = self.build_prompt(query, retrieved_chunks)
        
        # streaming version - yields tokens one by one
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.1,
            stream=True  # this enables streaming
        )
        
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token is not None:
                yield token


# TEST
if __name__ == "__main__":
    from src.retriever import Retriever
    
    retriever = Retriever()
    generator = Generator()
    
    query = "What happens if eBay suspends my account?"
    
    print(f"Query: {query}")
    print("\nRetrieving relevant chunks...")
    chunks = retriever.retrieve(query)
    
    print("\nGenerating answer...\n")
    print("--- ANSWER ---")
    
    # test streaming
    for token in generator.generate_stream(query, chunks):
        print(token, end="", flush=True)
    
    print("\n\n--- DONE ---")