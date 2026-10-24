"""Tiny test app for the graph component: streamlit run scripts/graph_component_demo.py."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ui.graph_component import graph_view  # noqa: E402

COLORS = ["#4e79a7", "#59a14f", "#f28e2b"]


def sample_payload() -> dict:
    """A 12-node, 3-community payload with a highlighted path."""
    names = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA", "IOTA", "KAPPA", "LAMBDA", "MU"]
    nodes = [{"id": n, "label": n, "type": "ORG", "community": str(i % 3), "degree": 1 + (i % 4),
              "description": f"{n} is a sample entity used to test the component. " * 3,
              "sources": [{"doc": f"doc_{i % 4}.txt", "excerpt": f"{n} appears in this excerpt."}]}
             for i, n in enumerate(names)]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (0, 6), (2, 9)]
    edges = [{"id": str(i), "source": names[a], "target": names[b], "label": "rel",
              "description": f"{names[a]} relates to {names[b]}", "sources": []} for i, (a, b) in enumerate(pairs)]
    return {"nodes": nodes, "edges": edges,
            "highlight": {"nodes": ["ALPHA", "BETA", "GAMMA", "DELTA"], "edges": ["0", "1", "2"],
                          "order": ["ALPHA", "BETA", "GAMMA", "DELTA"]},
            "communities": [{"id": str(i), "title": f"Community {i}", "color": COLORS[i], "size": 4} for i in range(3)],
            "options": {"show_full": True, "depth": 1, "pinned": []}}


st.set_page_config(page_title="Graph component test", layout="wide")
if "nonce" not in st.session_state:
    st.session_state.nonce = 0
if st.button("Focus ETA + THETA"):
    st.session_state.nonce += 1
event = graph_view(sample_payload(), {"nodes": ["ETA", "THETA"], "edges": [], "nonce": st.session_state.nonce},
                   key="demo", height=560)
st.write("Last event:", event)
