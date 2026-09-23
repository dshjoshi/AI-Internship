"""Ingest every CSV in a directory via POST /ingest, one file = one document,
each data row (excluding the header) = one chunk, sent as a pre-made `chunks`
list so no character-splitting happens.

Usage:
  python ingest_directory.py /path/to/dir --skip products.txt
  python ingest_directory.py /path/to/dir --skip products.txt --api-base-url https://ai-bootcamp-1.onrender.com
"""

import argparse
import os

import httpx


def file_to_chunks(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    data_rows = [line for line in lines[1:] if line]  # skip header, drop blank lines
    return data_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Directory of CSV files to ingest")
    parser.add_argument("--skip", nargs="*", default=[], help="Filenames to skip")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    filenames = sorted(
        f for f in os.listdir(args.directory)
        if f.endswith(".txt") and f not in args.skip
    )
    print(f"Files to ingest: {filenames}")

    with httpx.Client(timeout=600.0) as client:
        for filename in filenames:
            path = os.path.join(args.directory, filename)
            document_id = os.path.splitext(filename)[0]
            chunks = file_to_chunks(path)

            response = client.post(
                f"{args.api_base_url}/ingest",
                json={"document_id": document_id, "chunks": chunks, "source": filename},
            )
            response.raise_for_status()
            chunks_indexed = response.json()["chunks_indexed"]
            print(f"  {filename}: document_id={document_id!r} chunks_indexed={chunks_indexed}")

        health = client.get(f"{args.api_base_url}/health/pinecone").json()
        print(f"\nTotal vectors now in index {health['index']!r}: {health['total_vectors']}")


if __name__ == "__main__":
    main()
