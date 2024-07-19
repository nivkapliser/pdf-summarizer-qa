from sentence_transformers import SentenceTransformer # using pretrained model

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, chunks):
        return self.model.encode(chunks)

if __name__ == "__main__":
    # For testing purposes
    generator = EmbeddingGenerator()
    test_chunks = ["This is a test sentence.", "Another test sentence."]
    embeddings = generator.generate_embeddings(test_chunks)
    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")