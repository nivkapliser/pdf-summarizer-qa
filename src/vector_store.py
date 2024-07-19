import faiss
import numpy as np
from config import VECTOR_STORE_DIMENSION

class VectorStore:
    def __init__(self, dimension=VECTOR_STORE_DIMENSION):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add_vectors(self, vectors, texts):
        if len(vectors) != len(texts):
            raise ValueError("Number of vectors must match number of texts")

        self.index.add(np.array(vectors).astype('float32'))
        self.texts.extend(texts)

    def search(self, query_vector, k=5):
        query_vector = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        return [(self.texts[i], distances[0][j]) for j, i in enumerate(indices[0])]


if __name__ == "__main__":
    # For testing purposes
    store = VectorStore(384)  # Using the dimension from our embedding model
    test_vectors = np.random.rand(10, 384).astype('float32')
    test_texts = [f"Text {i}" for i in range(10)]
    store.add_vectors(test_vectors, test_texts)

    query = np.random.rand(384).astype('float32')
    results = store.search(query)
    print(f"Top 5 similar texts: {results}")