import numpy as np
from src.embedding_generator import EmbeddingGenerator


def test_embedding_generator():
    generator = EmbeddingGenerator()
    test_chunks = ["This is a test sentence.", "Another test sentence."]
    embeddings = generator.generate_embeddings(test_chunks)

    assert len(embeddings) == len(test_chunks)
    assert isinstance(embeddings[0], np.ndarray)
    assert embeddings[0].shape[0] == 384  # Dimension of 'all-MiniLM-L6-v2' embeddings