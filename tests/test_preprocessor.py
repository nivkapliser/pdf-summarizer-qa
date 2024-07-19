from src.preprocessor import preprocess_text

def test_preprocess_text():
    text = "Hello, World! This is a test. 123 How are you?"
    preprocessed = preprocess_text(text)
    assert len(preprocessed) == 3
    assert preprocessed[0] == "hello world"
    assert preprocessed[1] == "this is a test"
    assert preprocessed[2] == "how are you"