import openai
from src.embedding_generator import EmbeddingGenerator
from config import SUMMARY_PROMPT

class Summarizer:
    def __init__(self, vector_store, api_key):
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator()
        openai.api_key = api_key

    def summarize(self, query=SUMMARY_PROMPT, num_chunks=3):
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        relevant_chunks = self.vector_store.search(query_embedding, k=num_chunks)

        context = " ".join([chunk for chunk, _ in relevant_chunks])

        prompt = f"Summarize the following text in a concise manner:\n\n{context}\n\nSummary:"

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )

        return response.choices[0].text.strip()


if __name__ == "__main__":
    # For testing purposes
    from src.vector_store import VectorStore

    store = VectorStore(384)
    summarizer = Summarizer(store, "your-api-key-here")
    summary = summarizer.summarize("What is the main topic of this document?")
    print(f"Summary: {summary}")