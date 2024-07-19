from config import MAX_CHUNK_SIZE

# Splitting the text to chunks
def chunk_text(sentences, max_chunk_size=MAX_CHUNK_SIZE):
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for sentence in sentences:
        if current_chunk_size + len(sentence.split()) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_chunk_size = 0

        current_chunk.append(sentence)
        current_chunk_size += len(sentence.split())

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

