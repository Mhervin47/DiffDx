"""
One-time build script: embed each exemplar's context_summary with all-MiniLM-L6-v2
and write the float list back into the embedding field of each JSON in exemplars/.

Idempotent: exemplars that already have an embedding are skipped unless --force is passed.

Usage:
    python scripts/embed_exemplars.py
    python scripts/embed_exemplars.py --force   # re-embed all, even if populated
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentence_transformers import SentenceTransformer

from loop1.retrieval import load_pool, _EXEMPLARS_DIR, EMBED_MODEL_NAME


def main(force: bool = False) -> None:
    print(f"Loading model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    pool = load_pool(_EXEMPLARS_DIR)
    print(f"Loaded {len(pool)} exemplars from {_EXEMPLARS_DIR}")

    skipped = 0
    embedded = 0

    for exemplar in pool:
        json_path = _EXEMPLARS_DIR / f"{exemplar.exemplar_id}.json"

        if exemplar.embedding is not None and not force:
            print(f"  SKIP  {exemplar.exemplar_id} (already embedded)")
            skipped += 1
            continue

        vec = model.encode(exemplar.context_summary, normalize_embeddings=True)
        embedding_list: list[float] = vec.tolist()

        raw = json.loads(json_path.read_text(encoding="utf-8"))
        raw["embedding"] = embedding_list
        json_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  EMBED {exemplar.exemplar_id} ({len(embedding_list)}d)")
        embedded += 1

    print(f"\nDone. Embedded: {embedded}, Skipped: {skipped}")

    # Verify: reload and check all have embeddings
    pool_after = load_pool(_EXEMPLARS_DIR)
    missing = [ex.exemplar_id for ex in pool_after if ex.embedding is None]
    if missing:
        print(f"\nERROR: {len(missing)} exemplar(s) still have no embedding: {missing}", file=sys.stderr)
        sys.exit(1)
    else:
        dim = len(pool_after[0].embedding)  # type: ignore[arg-type]
        print(f"Verification passed: all {len(pool_after)} exemplars have {dim}d embeddings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed exemplar context_summary fields.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed even if embedding field is already populated.",
    )
    args = parser.parse_args()
    main(force=args.force)
