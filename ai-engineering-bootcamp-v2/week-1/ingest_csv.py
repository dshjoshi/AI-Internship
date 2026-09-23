"""Ingest a CSV file into the /ingest endpoint, one row per document/chunk.

Usage:
  python ingest_csv.py /path/to/products.txt
  python ingest_csv.py /path/to/products.txt --api-base-url https://ai-bootcamp-1.onrender.com

Each row is turned into one natural-language sentence (better for embeddings
than raw comma-separated values) and POSTed to /ingest as its own document,
so it lands as a single chunk with document_id="product-<productID>".
"""

import argparse
import csv
import os
import sys

import httpx


def row_to_text(row: dict) -> str:
    discontinued = "yes" if row["discontinued"] == "1" else "no"
    return (
        f"{row['productName']} (Product ID {row['productID']}): "
        f"packaged as {row['quantityPerUnit']}, priced at ${row['unitPrice']}, "
        f"supplied by supplier {row['supplierID']} in category {row['categoryID']}. "
        f"{row['unitsInStock']} units in stock, {row['unitsOnOrder']} on order, "
        f"reorder level {row['reorderLevel']}. Discontinued: {discontinued}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    source = os.path.basename(args.csv_path)
    succeeded = 0
    failed = 0
    chunks_indexed = 0

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Ingesting {len(rows)} rows from {args.csv_path} -> {args.api_base_url}/ingest")

    with httpx.Client(timeout=60.0) as client:
        for row in rows:
            document_id = f"product-{row['productID']}"
            payload = {
                "document_id": document_id,
                "text": row_to_text(row),
                "source": source,
            }
            try:
                response = client.post(f"{args.api_base_url}/ingest", json=payload)
                response.raise_for_status()
                chunks_indexed += response.json()["chunks_indexed"]
                succeeded += 1
                print(f"  ok  {document_id}")
            except httpx.HTTPError as exc:
                failed += 1
                print(f"  FAIL {document_id}: {exc}")

    print(f"\nDone. succeeded={succeeded} failed={failed} chunks_indexed={chunks_indexed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
