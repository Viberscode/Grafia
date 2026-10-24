"""Manual audit of GraphRAG extraction quality.

`python -m eval.extraction_audit` samples 30 relationships and 30 entities (seeded) into CSVs with empty
judgement columns; fill them with y/n, then `python -m eval.extraction_audit --score` computes precision.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from backend.engine import load_tables

logger = logging.getLogger("eval.extraction_audit")

RESULTS = Path(__file__).resolve().parent / "results"
REL_CSV = RESULTS / "extraction_audit.csv"
ENT_CSV = RESULTS / "extraction_audit_entities.csv"
PRECISION_JSON = RESULTS / "extraction_precision.json"
YES = {"y", "yes", "1", "true", "t", "correct"}
NO = {"n", "no", "0", "false", "f", "wrong", "incorrect"}
EXCERPT = 600


def _norm(title: str) -> str:
    """Lowercase alphanumeric words, used to spot duplicate entities."""
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def _chunk_text(text_unit_ids: object, chunks: dict[str, str]) -> str:
    """Text of the first source chunk, truncated."""
    ids = list(text_unit_ids) if text_unit_ids is not None else []
    return next((chunks[str(i)][:EXCERPT] for i in ids if str(i) in chunks), "")


def possible_duplicates(titles: list[str]) -> dict[str, list[str]]:
    """Other titles whose normalised form contains, or is contained in, this one."""
    normed = {t: _norm(t) for t in titles}
    return {t: [o for o in titles if o != t and n and normed[o] and (n in normed[o] or normed[o] in n)]
            for t, n in normed.items()}


def sample_relationships(tables: dict, n: int = 30, seed: int = 7) -> pd.DataFrame:
    """Seeded sample of relationships with their source chunk and an empty 'correct' column."""
    chunks = dict(zip(tables["text_units"]["id"].astype(str), tables["text_units"]["text"].astype(str)))
    rel = tables["relationships"]
    rows = rel.sample(n=min(n, len(rel)), random_state=seed)
    return pd.DataFrame({
        "id": rows["human_readable_id"].values,
        "subject": rows["source"].values,
        "relation": rows["description"].values,
        "object": rows["target"].values,
        "source_chunk": [_chunk_text(x, chunks) for x in rows["text_unit_ids"]],
        "correct": "",
    })


def sample_entities(tables: dict, n: int = 30, seed: int = 7) -> pd.DataFrame:
    """Seeded sample of entities with type, description, likely duplicates and empty judgement columns."""
    chunks = dict(zip(tables["text_units"]["id"].astype(str), tables["text_units"]["text"].astype(str)))
    ent = tables["entities"]
    dups = possible_duplicates(ent["title"].astype(str).tolist())
    rows = ent.sample(n=min(n, len(ent)), random_state=seed)
    return pd.DataFrame({
        "id": rows["human_readable_id"].values,
        "title": rows["title"].values,
        "type": rows["type"].values,
        "description": [str(d)[:300] for d in rows["description"]],
        "possible_duplicates": ["; ".join(dups.get(str(t), [])) for t in rows["title"]],
        "source_chunk": [_chunk_text(x, chunks) for x in rows["text_unit_ids"]],
        "type_correct": "",
        "duplicate": "",
    })


def _judgements(series: pd.Series) -> list[int]:
    """Parse y/n style cells into 1/0, ignoring blanks and unknown values."""
    out = []
    for v in series.fillna("").astype(str).str.strip().str.lower():
        if v in YES:
            out.append(1)
        elif v in NO:
            out.append(0)
    return out


def _rate(values: list[int]) -> float | None:
    """Mean of 0/1 judgements, or None when nothing is filled in."""
    return round(sum(values) / len(values), 3) if values else None


def score(rel_csv: Path = REL_CSV, ent_csv: Path = ENT_CSV) -> dict:
    """Relationship precision, entity type precision and duplicate rate from the filled CSVs."""
    out: dict[str, object] = {}
    if rel_csv.exists():
        j = _judgements(pd.read_csv(rel_csv, dtype=str)["correct"])
        out.update(relationship_precision=_rate(j), relationships_judged=len(j))
    if ent_csv.exists():
        df = pd.read_csv(ent_csv, dtype=str)
        t, d = _judgements(df["type_correct"]), _judgements(df["duplicate"])
        out.update(entity_type_precision=_rate(t), entities_judged=len(t), duplicate_rate=_rate(d))
    return out


def _has_judgements(path: Path, columns: list[str]) -> bool:
    """True when an existing audit file already has filled judgement cells."""
    if not path.exists():
        return False
    df = pd.read_csv(path, dtype=str)
    return any(_judgements(df[c]) for c in columns if c in df.columns)


def main() -> None:
    """Write the audit samples, or score them with --score."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", action="store_true", help="compute precision from the filled CSVs")
    ap.add_argument("--n", type=int, default=30, help="sample size for relationships and entities")
    ap.add_argument("--seed", type=int, default=7, help="random seed for the sample")
    ap.add_argument("--force", action="store_true", help="overwrite audit files that already have judgements")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.score:
        result = score()
        if not result:
            logger.error("No audit files yet: run `python -m eval.extraction_audit` first.")
            sys.exit(1)
        if not result.get("relationships_judged") and not result.get("entities_judged"):
            logger.error("No judgements yet: fill 'correct', 'type_correct' and 'duplicate' with y/n first.")
            sys.exit(1)
        PRECISION_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        return

    if not args.force and (_has_judgements(REL_CSV, ["correct"])
                           or _has_judgements(ENT_CSV, ["type_correct", "duplicate"])):
        logger.error("Audit files already contain judgements; use --score, or --force to resample.")
        sys.exit(1)
    tables = load_tables()
    if not {"relationships", "entities", "text_units"} <= set(tables):
        logger.error("No index found: build the knowledge graph first.")
        sys.exit(1)
    sample_relationships(tables, args.n, args.seed).to_csv(REL_CSV, index=False)
    sample_entities(tables, args.n, args.seed).to_csv(ENT_CSV, index=False)
    logger.info("Wrote %s and %s; fill 'correct', 'type_correct' and 'duplicate' with y/n, then run --score",
                REL_CSV, ENT_CSV)


if __name__ == "__main__":
    main()
