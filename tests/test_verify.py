"""Tests for backend.verify and backend.abstain (offline: support check is stubbed)."""
from backend import verify
from backend.abstain import should_abstain
from backend.contracts import Claim, Evidence

CHUNKS = {
    "c1": "The Kestrel Battery Storage Project entered service in June 2024.",
    "c2": "Helix Turbines supplied 40 floating turbines.",
}


def test_fabricated_quote_dropped_real_quote_kept():
    claims = [
        {"text": "Kestrel started in June 2024.",
         "evidence": [{"chunk_id": "c1", "quote": "entered  service in June 2024"}]},
        {"text": "Kestrel cost a billion dollars.",
         "evidence": [{"chunk_id": "c1", "quote": "cost one billion dollars"}]},
        {"text": "Wrong chunk cited.",
         "evidence": [{"chunk_id": "c2", "quote": "entered service in June 2024"}]},
    ]
    kept = verify.verify_claims(claims, CHUNKS, support_check=False)
    assert [c.text for c in kept] == ["Kestrel started in June 2024."]


def test_mixed_evidence_keeps_only_valid_quotes():
    claims = [{"text": "Turbines.", "evidence": [
        {"chunk_id": "c2", "quote": "Helix Turbines supplied 40 floating turbines."},
        {"chunk_id": "c2", "quote": "Helix built 90 turbines"},
    ]}]
    kept = verify.check_quotes(claims, CHUNKS)
    assert len(kept[0].evidence) == 1


def test_support_check_drops_unsupported(monkeypatch):
    monkeypatch.setattr(verify, "chat_json", lambda *a, **k: {"results": [
        {"index": 0, "supported": True}, {"index": 1, "supported": False}]})
    claims = [Claim(text="a", evidence=[Evidence(chunk_id="c1", quote="x")]),
              Claim(text="b", evidence=[Evidence(chunk_id="c1", quote="y")])]
    assert [c.text for c in verify.check_support(claims)] == ["a"]


def test_abstain_rules():
    claim = Claim(text="a", evidence=[Evidence(chunk_id="c1", quote="x")])
    assert should_abstain("abstain", {}, [])[0]
    assert should_abstain("local", {"context": 0, "signal": 3}, [claim])[0]
    assert should_abstain("local", {"context": 3, "signal": 0}, [claim])[0]
    assert should_abstain("global", {"context": 3, "signal": 2}, [])[0]
    assert should_abstain("global", {"context": 3, "signal": 2}, [claim]) == (False, None)
    assert "Energy Topic" in should_abstain("abstain", {}, [], "Energy Topic")[1]
