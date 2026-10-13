"""Retrieval eval: all keywords must appear in top-k retrieved text.
Usage: python -m eval.run_eval [site_id]
"""
import json
import sys
from pathlib import Path

from app.rag_engine.retrieve import retrieve

questions = json.loads((Path(__file__).parent / "eval_questions.json").read_text(encoding="utf-8"))
site_id = sys.argv[1] if len(sys.argv) > 1 else "resiliencesoft"
passed = 0
for q in questions:
    text = " ".join(h["text"] for h in retrieve(q["question"], site_id, ["public"])).lower()
    missing = [k for k in q.get("keywords", []) if k.lower() not in text]
    passed += not missing
    print(("  ok  " if not missing else "  FAIL") , q["question"], missing or "")
print(f"\nHit rate: {passed}/{len(questions)}")
