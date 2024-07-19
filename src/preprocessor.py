import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = re.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Split into sentences
    sentences = sent_tokenize(text)

    return sentences