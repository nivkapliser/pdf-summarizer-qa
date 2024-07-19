import pytest
from unittest.mock import Mock, patch
from src.summarizer import Summarizer
from src.vector_store import VectorStore


@pytest.fixture
def mock_openai():
    with patch('openai.Completion.create') as mock:
        mock.return_value.choices[0].text = "This is a test summary."
        yield mock


def test_summarizer(mock_openai):
    store = VectorStore(384)
    summarizer = Summarizer(store, "test-api-key")

    # Mock the vector store search method
    store.search = Mock(return_value=[("This is a test chunk.", 0.9)])

    summary = summarizer.summarize("Test query")
    assert summary == "This is a test summary."
    mock_openai.assert_called_once()