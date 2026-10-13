"""Push repo knowledge into Atlas. Render runs this before the server starts.

Usage: python -m ingestion.sync
"""
from pathlib import Path

from ingestion.ingest import ingest

ROOT = Path(__file__).resolve().parent.parent / "data"
# ponytail: add a row when a new product folder lands in data/
JOBS = (
    ("resiliencesoft", "general", "https://www.resiliencesoft.com", ROOT),
    ("resiliohub", "resiliohub", "https://resiliohub.com/", ROOT / "resiliohub"),
    ("sharpquiz", "sharpquiz", "https://sharpquiz.com/", ROOT / "sharpquiz"),
)


def main():
    for site_id, product_id, url, folder in JOBS:
        chunks = folder / "chunks.jsonl"
        if not chunks.exists():
            raise SystemExit(f"missing {chunks}")
        print(f"== {site_id}")
        ingest(site_id, product_id, url, folder)


if __name__ == "__main__":
    main()
