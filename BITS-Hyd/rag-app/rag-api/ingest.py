"""
Simple ingestion script: chunk text/HTML files and upsert them into Qdrant.

Usage (run from inside the rag-api pod, or locally against a port-forwarded
Qdrant):

    python ingest.py --docs-dir ./docs

Each .txt/.md/.html file under --docs-dir is split into overlapping chunks,
embedded with the same sentence-transformers model used at query time, and
stored in the configured Qdrant collection.

For .html files, only the page's visible text is extracted (scripts,
styles, nav/header/footer boilerplate, and markup are stripped) — saved
webpage asset folders like "<page>_files/" are ignored since we only glob
for *.html itself.
"""

import argparse
import os
import uuid
from pathlib import Path

from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def extract_html_text(html: str) -> str:
    """Strip an HTML page down to its visible, readable text."""
    soup = BeautifulSoup(html, "html.parser")

    # Drop elements that never contain meaningful article text
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "svg"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse blank/whitespace-only lines
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def read_file_text(f: Path) -> str:
    if f.suffix.lower() in (".html", ".htm"):
        raw = f.read_text(encoding="utf-8", errors="ignore")
        return extract_html_text(raw)
    return f.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs-dir",
        default="./docs",
        help="Directory of .txt/.md/.html files to ingest (default: ./docs)",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection first")
    args = parser.parse_args()

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vector_size = embedder.get_sentence_embedding_dimension()

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if args.recreate or not client.collection_exists(QDRANT_COLLECTION):
        print(f"Creating collection '{QDRANT_COLLECTION}' (dim={vector_size})")
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    docs_dir = Path(args.docs_dir)
    files = (
        list(docs_dir.rglob("*.txt"))
        + list(docs_dir.rglob("*.md"))
        + list(docs_dir.rglob("*.html"))
        + list(docs_dir.rglob("*.htm"))
        + list(docs_dir.rglob("*.doc"))
        + list(docs_dir.rglob("*.pdf"))
        + list(docs_dir.rglob("*.docx"))
        + list(docs_dir.rglob("*.csv"))
        + list(docs_dir.rglob("*.xls"))
        + list(docs_dir.rglob("*.xlsx"))
        )
    if not files:
        print(f"No .txt, .md, .html,  .pdf, .doc, .docx, .csv, .xlsx or .htm files found under {docs_dir}")
        return

    points = []
    for f in files:
        text = read_file_text(f)
        if not text.strip():
            print(f"  skipping {f} (no extractable text)")
            continue
        print(f"  processing {f} ({len(text)} chars extracted)")
        for chunk in chunk_text(text, args.chunk_size, args.overlap):
            vector = embedder.encode(chunk).tolist()
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"text": chunk, "source": str(f.relative_to(docs_dir))},
                )
            )

    print(f"Upserting {len(points)} chunks from {len(files)} files...")
    # Upsert in batches to keep request sizes reasonable
    batch_size = 128
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i + batch_size])

    print("Done.")


if __name__ == "__main__":
    main()
