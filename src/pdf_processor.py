import fitz
import re


def clean_text(text):
    # Remove unwanted labels while keeping the actual answer
    text = re.sub(
        r"(?i)\bViva\s+answer\s*:\s*",
        "",
        text
    )

    text = re.sub(
        r"(?i)\bEasy\s+explanation\s*:\s*",
        "",
        text
    )

    text = re.sub(
        r"(?i)\bRemember\s*:\s*",
        "",
        text
    )

    # Fix common PDF character-mapping issues
    text = text.replace("ϑ", "ti")
    text = text.replace("θ", "ti")
    text = text.replace("Θ", "ti")
    text = text.replace("☒", "•")
    text = text.replace("□", "•")

    return text


def extract_text_from_pdf(uploaded_file):
    pdf = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(pdf, start=1):
        text = clean_text(page.get_text())

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    pdf.close()
    return pages


def split_text(pages, source_name, chunk_size=700):
    chunks = []
    chunk_id = 0

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]

        # Separate common headings before splitting
        text = re.sub(
            r"(?i)\s+(?=(?:Q\d+\s*[.:]|"
            r"\d{1,2}\.\s+[A-Z]|"
            r"Role/Job Title:|Job Purpose:|"
            r"Roles & Responsibilities:))",
            "\n",
            text
        )

        chunks_to_process = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            # Split long lines at sentence boundaries
            sentences = re.split(
                r"(?<=[.!?])\s+",
                line
            )

            for sentence in sentences:
                sentence = sentence.strip()

                if not sentence:
                    continue

                # Keep even long sentences within the chunk limit
                words = sentence.split()
                piece = ""

                for word in words:
                    if piece and len(piece) + len(word) + 1 > chunk_size:
                        chunks_to_process.append(piece)
                        piece = word
                    else:
                        piece = f"{piece} {word}".strip()

                if piece:
                    chunks_to_process.append(piece)

        current_chunk = ""

        for piece in chunks_to_process:
            if (
                current_chunk
                and len(current_chunk) + len(piece) + 1 > chunk_size
            ):
                chunks.append({
                    "chunk_id": chunk_id,
                    "source": source_name,
                    "page": page_number,
                    "text": current_chunk.strip()
                })

                chunk_id += 1
                current_chunk = piece

            else:
                current_chunk = (
                    f"{current_chunk}\n{piece}".strip()
                )

        if current_chunk.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "source": source_name,
                "page": page_number,
                "text": current_chunk.strip()
            })

            chunk_id += 1

    return chunks