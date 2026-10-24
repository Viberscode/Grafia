"""Decide when the system should decline to answer."""
from __future__ import annotations

from backend.config import ABSTAIN_MIN_SCORE
from backend.contracts import Claim


def should_abstain(
    route: str,
    scores: dict[str, int],
    claims: list[Claim],
    closest_community: str | None = None,
) -> tuple[bool, str | None]:
    """Return (abstain, reason) from scores {"context": chunks, "signal": entities or reports}."""
    hint = f" The closest topic in the corpus is \u201c{closest_community}\u201d." if closest_community else ""
    if route == "abstain":
        return True, f"This question looks outside the scope of the indexed documents.{hint}"
    if scores.get("context", 0) <= 0:
        return True, f"No relevant passages were retrieved for this question.{hint}"
    if scores.get("signal", 0) < ABSTAIN_MIN_SCORE:
        what = "community reports" if route == "global" else "matching entities"
        return True, f"Too few {what} were found to answer reliably.{hint}"
    if not claims:
        return True, f"No statement could be backed by a verified quote from the sources.{hint}"
    return False, None
