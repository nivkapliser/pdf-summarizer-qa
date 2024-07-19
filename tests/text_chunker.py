from src.chunker import chunk_text

def test_chunk_text():
    sentences = [
        "This is the first sentence.",
        "Here's the second one.",
        "And the third.",
        "Fourth sentence is here.",
        "Fifth sentence is the last in this test."
    ]
    chunks = chunk_text(sentences, max_chunk_size=10)
    assert len(chunks) == 3
    assert chunks[0] == "This is the first sentence. Here's the second one."
    assert chunks[1] == "And the third. Fourth sentence is here."
    assert chunks[2] == "Fifth sentence is the last in this test."