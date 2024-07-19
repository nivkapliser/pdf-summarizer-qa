import PyPDF2

def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()

    return text

if __name__ == '__main__':
    # For test
    sample_text = extract_text_from_pdf('../data/sample_pdf.pdf')
    print(sample_text[:500]) # print's the first 500 characters

