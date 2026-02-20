import json
from pathlib import Path

from app.db import Base, SessionLocal, engine
from app.models import DocumentChunk
from app.rag import RAGService

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk.strip()]


def ingest_directory(data_dir: Path):
    Base.metadata.create_all(bind=engine)
    rag = RAGService()
    db = SessionLocal()

    try:
        db.query(DocumentChunk).delete()
        db.commit()

        files = [
            p
            for p in data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".txt", ".md", ".csv", ".json", ".pdf"}
        ]

        processed = 0
        for file_path in files:
            suffix = file_path.suffix.lower()
            text = ""
            if suffix == ".pdf":
                if PdfReader is None:
                    print("Skipping PDF ingestion: PyPDF2 is not installed.")
                    continue
                try:
                    reader = PdfReader(str(file_path))
                    pages = []
                    for p in reader.pages:
                        try:
                            pages.append(p.extract_text() or "")
                        except Exception:
                            pages.append("")
                    text = "\n".join(pages)
                except Exception as e:
                    print(f"Failed to read PDF {file_path}: {e}")
                    continue
            else:
                text = file_path.read_text(encoding="utf-8", errors="ignore")

            if not text or not text.strip():
                # nothing to index for this file
                continue

            for chunk in split_text(text):
                emb = rag.embed_text(chunk)
                row = DocumentChunk(
                    source=str(file_path.relative_to(data_dir.parent)),
                    chunk_text=chunk,
                    embedding_json=json.dumps(emb),
                )
                db.add(row)
            processed += 1

        db.commit()
        print(f"Ingestion complete. Indexed {processed} files.")
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest files from data directory into embeddings table.")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    ingest_directory(Path(args.data_dir).resolve())
