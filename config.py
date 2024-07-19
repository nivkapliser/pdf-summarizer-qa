import os

# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Vector store configuration
VECTOR_STORE_DIMENSION = 384

# Chunking configuration
MAX_CHUNK_SIZE = 100

# Summarization configuration
SUMMARY_PROMPT = "Provide a brief summary of the main points in this document."