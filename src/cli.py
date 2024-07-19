import argparse
import os
from src.pdf_extractor import extract_text_from_pdf
from src.preprocessor import preprocess_text
from src.chunker import chunk_text
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore
from src.summarizer import Summarizer
from src.qa_system import QASystem


def main():
    parser = argparse.ArgumentParser(description="PDF Summarizer and QA System")
    parser.add_argument("file_path", help="Path to the PDF file")
    parser.add_argument("--summarize", action="store_true", help="Generate a summary of the document")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive QA mode")
    args = parser.parse_args()

    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")

        extracted_text = extract_text_from_pdf(args.file_path)
        preprocessed_sentences = preprocess_text(extracted_text)
        chunks = chunk_text(preprocessed_sentences, max_chunk_size=100)

        generator = EmbeddingGenerator()
        embeddings = generator.generate_embeddings(chunks)

        vector_store = VectorStore(dimension=384)
        vector_store.add_vectors(embeddings, chunks)

        summarizer = Summarizer(vector_store, api_key)
        qa_system = QASystem(vector_store, api_key)

        if args.summarize:
            summary = summarizer.summarize("Provide a brief summary of the main points in this document.")
            print(f"Summary of the document:\n{summary}\n")

        if args.interactive:
            print("Entering interactive QA mode. Type 'quit' to exit.")
            while True:
                question = input("Ask a question about the document: ")
                if question.lower() == 'quit':
                    break
                answer = qa_system.answer_question(question)
                print(f"Answer: {answer}\n")

    except FileNotFoundError:
        print(f"Error: The file '{args.file_path}' was not found.")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()