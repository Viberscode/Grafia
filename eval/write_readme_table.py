"""Replace the README results placeholder with eval/results/summary.md (and extraction precision if scored)."""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

from eval.extraction_audit import PRECISION_JSON
from eval.run import RESULTS

logger = logging.getLogger("eval.write_readme_table")

README = Path(__file__).resolve().parent.parent / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def extraction_line(path: Path = PRECISION_JSON) -> str:
    """One sentence with the audited extraction precision, or '' if not scored yet."""
    if not path.exists():
        return ""
    p = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    if p.get("relationship_precision") is not None:
        parts.append(f"relationship precision {p['relationship_precision']:.2f} ({p['relationships_judged']} judged)")
    if p.get("entity_type_precision") is not None:
        parts.append(f"entity type precision {p['entity_type_precision']:.2f} ({p['entities_judged']} judged)")
    if p.get("duplicate_rate") is not None:
        parts.append(f"duplicate entity rate {p['duplicate_rate']:.2f}")
    return ("Extraction audit (`python -m eval.extraction_audit --score`): " + ", ".join(parts) + ".") if parts else ""


def replace_block(readme: str, body: str) -> str:
    """Swap the text between the RESULTS markers for body."""
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(readme):
        raise ValueError(f"README has no {START} ... {END} placeholder")
    return pattern.sub(lambda _: f"{START}\n{body.strip()}\n{END}", readme, count=1)


def main() -> None:
    """Write the generated tables into README.md."""
    summary = RESULTS / "summary.md"
    if not summary.exists():
        logger.error("Run `python -m eval.report` first to create %s.", summary)
        sys.exit(1)
    body = summary.read_text(encoding="utf-8")
    line = extraction_line()
    if line:
        body = f"{body.strip()}\n\n{line}"
    README.write_text(replace_block(README.read_text(encoding="utf-8"), body), encoding="utf-8")
    logger.info("Updated the results section of %s", README)


if __name__ == "__main__":
    main()
