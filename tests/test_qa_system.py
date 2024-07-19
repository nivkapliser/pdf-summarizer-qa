import pytest
from unittest.mock import Mock, patch
from src.qa_system import QASystem
from src.vector_store import VectorStore


@pytest.fixture
def mock_openai():
    with patch('openai.Completion.create') as mock:
        mock.return_value.choices[0].text = "This is a test answer."
        yield mock


def test_qa_system(mock_openai):
    store = VectorStore(384)
    qa_system = QASystem(store, "test-api-key")

    # Mock the vector store search method
    store.search = Mock(return_value=[("This is a test chunk.", 0.9)])

    answer = qa_system.answer_question("Test question?")
    assert answer == "This is a test answer."
    mock_openai.assert_called_once()