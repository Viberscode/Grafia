"""The recorded demo seed parses into Responses and the flag is detected."""
from __future__ import annotations

from ui import demo_seed


def test_demo_seed_file_has_three_grounded_answers() -> None:
    seeds = demo_seed.load_seed()
    assert [q for q, _, _ in seeds] == [q for q, _ in demo_seed.QUESTIONS]
    assert {r.route for _, _, r in seeds} == {"global", "local", "hybrid"}
    assert all(r.claims and r.trace and not r.abstained for _, _, r in seeds)


def test_demo_seed_flag(monkeypatch) -> None:
    monkeypatch.delenv("DEMO_SEED", raising=False)
    assert demo_seed.requested(["app.py", "--demo-seed"])
    assert not demo_seed.requested(["app.py"])
    monkeypatch.setenv("DEMO_SEED", "1")
    assert demo_seed.requested(["app.py"])
