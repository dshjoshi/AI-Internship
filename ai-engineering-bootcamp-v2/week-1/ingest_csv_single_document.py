"""Ingest a CSV file as ONE document, with each row as its own chunk.

Unlike ingest_csv.py (one row = one document, via the /ingest HTTP endpoint
and its character-based splitter), this treats the whole file as a single
document_id and hands each row's text straight to upsert_chunks() as an
already-made chunk — so chunk boundaries always align exactly with rows,
regardless of CHUNK_SIZE.

Calls vectorstore.py directly (no running server needed) and targets a
separate index via --index-name, so it never touches the index your live
app is currently configured for.

Usage:
  python ingest_csv_single_document.py /path/to/products.txt --index-name products-single-doc
"""

import argparse
import csv
import os

from ingest_csv import row_to_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("--index-name", required=True, help="Pinecone index to upsert into")
    args = parser.parse_args()

    # Set before importing vectorstore/load_dotenv so this run targets --index-name
    # without touching .env or any already-running server's configuration.
    os.environ["PINECONE_INDEX_NAME"] = args.index_name

    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))  # fills in API keys only

    from vectorstore import upsert_chunks

    source = os.path.basename(args.csv_path)
    document_id = os.path.splitext(source)[0]

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    chunks = [row_to_text(row) for row in rows]

    print(f"Ingesting {len(chunks)} rows from {args.csv_path} as document_id={document_id!r} "
          f"into index {args.index_name!r}")

    chunks_indexed = upsert_chunks(document_id, chunks, source)

    print(f"\nDone. document_id={document_id} chunks_indexed={chunks_indexed}")


if __name__ == "__main__":
    main()
