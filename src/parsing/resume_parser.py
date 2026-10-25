
"""Extract text from PDF and DOCX resume files."""

from pathlib import Path


def extract_resume_text(file_path):
    """Extract text from a PDF or DOCX resume."""

    path = Path(file_path)
    extension = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    if extension == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]
        text = "\n".join(pages)

    elif extension == ".docx":
        from docx import Document

        document = Document(str(path))
        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    elif extension == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")

    else:
        raise ValueError(
            "Unsupported file type. Use PDF, DOCX, or TXT."
        )

    return text.strip()


if __name__ == "__main__":
    print("Resume parser is ready!")
    print("Supported formats: PDF, DOCX, TXT")
