#!/usr/bin/env python3
"""
build_chunks.py — Build chunks.jsonl from knowledge_base.md and qa_pairs.json.

Reads the markdown knowledge base and structured Q&A pairs,
splits them into retrieval-ready chunks, and writes chunks.jsonl.

Long sections are split at roughly MAX_CHUNK_WORDS so that every chunk
fits inside the embedding model's token window (~256 tokens for MiniLM).

Usage:
    python -m ingestion.build_chunks
"""

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"

MAX_CHUNK_WORDS = 300  # split KB sections longer than this


# ---------------------------------------------------------------------------
# Markdown knowledge base parser
# ---------------------------------------------------------------------------

def _split_long(text: str, limit: int = MAX_CHUNK_WORDS) -> list[str]:
    """
    Split *text* into pieces of at most *limit* words.

    Tries to break on blank lines first (paragraph boundaries),
    then on any newline, and only as a last resort on word count.
    """
    words = text.split()
    if len(words) <= limit:
        return [text]

    # Try paragraph-level split first
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        parts: list[str] = []
        buf: list[str] = []
        buf_words = 0
        for para in paragraphs:
            pw = len(para.split())
            if buf and buf_words + pw > limit:
                parts.append("\n\n".join(buf))
                buf = [para]
                buf_words = pw
            else:
                buf.append(para)
                buf_words += pw
        if buf:
            parts.append("\n\n".join(buf))
        return parts

    # Line-level split
    lines = text.splitlines()
    parts = []
    buf = []
    buf_words = 0
    for line in lines:
        lw = len(line.split())
        if buf and buf_words + lw > limit:
            parts.append("\n".join(buf))
            buf = [line]
            buf_words = lw
        else:
            buf.append(line)
            buf_words += lw
    if buf:
        parts.append("\n".join(buf))
    return parts


def parse_knowledge_base(path: Path, brand: str = "ResilienceSoft") -> list[dict]:
    """
    Parse knowledge_base.md into chunks.

    Rules:
    - Each H2 (##) starts a new section.
    - Each H3 (###) within a section starts a sub-chunk.
    - The "Gaps and inconsistencies" section is tagged visibility: internal.
    - Sections longer than MAX_CHUNK_WORDS are split further.
    """
    if not path.exists():
        print(f"⚠  {path} not found — skipping knowledge base chunks.")
        return []

    text = path.read_text(encoding="utf-8")
    chunks: list[dict] = []
    chunk_id = 0
    current_section = ""
    current_sub = ""
    current_lines: list[str] = []
    visibility = "public"
    doc_title = ""
    current_source = ""

    def flush():
        nonlocal chunk_id
        body = "\n".join(current_lines).strip()
        if not body:
            return
        section = current_section or doc_title or "Introduction"
        sub_label = f" > {current_sub}" if current_sub else ""
        title = f"{brand} - {section}{sub_label}"
        meta_section = f"{section}{sub_label}"

        parts = _split_long(body)
        for idx, part in enumerate(parts):
            chunk_id += 1
            suffix = f" (part {idx + 1})" if len(parts) > 1 else ""
            chunks.append({
                "id": f"kb-{chunk_id:03d}",
                "text": f"{title}{suffix}\n{part.strip()}",
                "metadata": {
                    "source": "website",
                    "section": meta_section,
                    "type": "doc",
                    "visibility": visibility,
                    **({"source_url": current_source} if current_source else {}),
                },
            })

    for line in text.splitlines():
        if line.startswith("> Source: ") and not current_lines:
            # per-section canonical URL; kept out of the chunk text
            current_source = line[len("> Source: "):].strip().split()[0]
            continue
        if line.startswith("# ") and not line.startswith("## "):
            # H1 — document title; treat as section name for preamble
            flush()
            doc_title = line[2:].strip()
            current_section = doc_title
            current_sub = ""
            current_lines = []
        elif line.startswith("## "):
            flush()
            current_source = ""
            current_section = line[3:].strip()
            current_sub = ""
            current_lines = []
            # Mark "gaps" section as internal
            visibility = (
                "internal"
                if "gap" in current_section.lower()
                or "inconsistenc" in current_section.lower()
                else "public"
            )
        elif line.startswith("### "):
            flush()
            current_sub = line[4:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Q&A pairs
# ---------------------------------------------------------------------------

def parse_qa_pairs(path: Path) -> list[dict]:
    """Load qa_pairs.json and create one chunk per pair."""
    if not path.exists():
        print(f"⚠  {path} not found — skipping Q&A chunks.")
        return []

    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)

    chunks = []
    for i, pair in enumerate(pairs, start=1):
        q = pair["question"]
        a = pair["answer"]
        chunks.append({
            "id": f"qa-{i:03d}",
            "text": f"Q: {q}\nA: {a}",
            "metadata": {
                "source": "qa_pairs",
                "section": "Q&A",
                "type": "qa",
                "visibility": pair.get("visibility", "public"),
                **({"source_url": pair["source_url"]} if pair.get("source_url") else {}),
            },
        })
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT), help="folder with knowledge_base.md and qa_pairs.json")
    ap.add_argument("--brand", default="ResilienceSoft")
    a = ap.parse_args()
    base = Path(a.dir)
    OUT_FILE = base / "chunks.jsonl"
    kb_chunks = parse_knowledge_base(base / "knowledge_base.md", a.brand)
    qa_chunks = parse_qa_pairs(base / "qa_pairs.json")
    all_chunks = kb_chunks + qa_chunks

    if not all_chunks:
        print("Nothing to write. Supply knowledge_base.md and/or qa_pairs.json.")
        return

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Stats
    word_counts = [len(c["text"].split()) for c in all_chunks]
    public = [c for c in all_chunks if c["metadata"]["visibility"] == "public"]
    internal = [c for c in all_chunks if c["metadata"]["visibility"] == "internal"]

    print(f"✅ Wrote {len(all_chunks)} chunks to {OUT_FILE}")
    print(f"   {len(kb_chunks)} knowledge-base chunks, {len(qa_chunks)} Q&A chunks")
    print(f"   {len(public)} public, {len(internal)} internal")
    print(f"   Words: avg {statistics.mean(word_counts):.0f}, "
          f"max {max(word_counts)}, min {min(word_counts)}")


if __name__ == "__main__":
    main()
