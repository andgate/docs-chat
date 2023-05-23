import re

from PyPDF2 import PdfReader


def preprocess_pdf():
    src_path = "assets/1240-S.PL.pdf"
    dest_path = "assets/1240-S.PL.txt"

    pdf_reader = PdfReader(src_path)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        cleaned_text = re.sub(r"\s*\d+\s*$", "", page_text, flags=re.MULTILINE)
        text += cleaned_text

    with open(dest_path, "w") as text_file:
        text_file.write(text)
