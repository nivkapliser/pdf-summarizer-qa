import openai
from src.embedding_generator import EmbeddingGenerator


class QASystem:
    def __init__(self, vector_store, api_key):
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator()
        openai.api_key = api_key

    def answer_question(self, question, num_chunks=3):
        question_embedding = self.embedding_generator.generate_embeddings([question])[0]
        relevant_chunks = self.vector_store.search(question_embedding, k=num_chunks)

        context = " ".join([chunk for chunk, _ in relevant_chunks])

        prompt = f"Answer the following question based on the given context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()


if __name__ == "__main__":
    # For testing purposes
    from src.vector_store import VectorStore

    store = VectorStore(384)
    qa_system = QASystem(store, "your-api-key-here")
    answer = qa_system.answer_question("What is the main topic of this document?")
    print(f"Answer: {answer}")