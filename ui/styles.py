"""Global CSS, route badge metadata and the Guide popover."""
from __future__ import annotations

import streamlit as st

ROUTES = {
    "global": {"icon": ":material/hub:", "color": "violet", "label": "Global (themes)"},
    "local": {"icon": ":material/my_location:", "color": "blue", "label": "Local (entities)"},
    "basic": {"icon": ":material/search:", "color": "gray", "label": "Basic (vector)"},
    "abstain": {"icon": ":material/block:", "color": "orange", "label": "Abstained"},
}

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1600px; }
.tg-card { border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: .8rem 1rem; margin: .35rem 0;
           background: rgba(128,128,128,.06); }
.tg-claim { border-left: 3px solid #22a06b; }
.tg-abstain { border-left: 3px solid #e8a33d; background: rgba(232,163,61,.08); }
.tg-muted { opacity: .72; font-size: .86rem; }
.tg-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .2rem 0 .4rem; }
.tg-chip { border: 1px solid rgba(128,128,128,.35); border-radius: 999px; padding: .12rem .65rem; font-size: .8rem;
           background: rgba(128,128,128,.08); }
.tg-chip b { font-weight: 700; }
.tg-status-ok { color: #22a06b; } .tg-status-demo { color: #e8a33d; }
.tg-quote { border-left: 3px solid rgba(128,128,128,.45); padding: .2rem .7rem; margin: .3rem 0; font-style: italic; }
.tg-chunk { white-space: pre-wrap; line-height: 1.5; font-size: .9rem; }
.tg-chunk mark, .tg-diff mark { background: rgba(255,200,0,.38); color: inherit; padding: 0 2px; border-radius: 3px; }
.tg-diff mark.only { background: rgba(34,160,107,.30); }
.tg-dot { display: inline-block; width: .7rem; height: .7rem; border-radius: 50%; margin-right: .35rem; vertical-align: middle; }
[data-testid="stColumn"]:has(.st-key-tg_right) { position: sticky; top: 3.4rem; align-self: flex-start; }
[data-testid="stHorizontalBlock"]:has(.st-key-tg_right) { align-items: flex-start; }
.st-key-tg_chat [data-testid="stChatMessage"] { border-radius: 14px; }
[class*="st-key-tg_claim_"] { border-left: 4px solid #22a06b !important; border-radius: 12px !important; }
[class*="st-key-tg_abstain_"] { border-left: 4px solid #e8a33d !important; background: rgba(232,163,61,.08);
                                border-radius: 12px !important; }
[class*="st-key-tg_theme_card_"] { border-radius: 14px !important; }
.st-key-tg_followups button p { font-size: .82rem; }
@media (max-width: 900px) {
  [data-testid="stHorizontalBlock"]:has(.st-key-tg_right) { flex-direction: column; }
  [data-testid="stHorizontalBlock"]:has(.st-key-tg_right) > [data-testid="stColumn"] {
    width: 100% !important; flex: 1 1 100% !important; position: static; }
}
</style>
"""

GUIDE = """
**Ask** a question in the chat. The router picks a strategy:

- :violet-badge[:material/hub: Global]: themes across the whole corpus (community reports)
- :blue-badge[:material/my_location: Local]: specific entities and how they connect
- :orange-badge[:material/block: Abstain]: out of scope, with the closest theme instead

Every claim is backed by a **verified verbatim quote**. Click **Show in graph** to zoom to its evidence.

**Graph**: red = evidence, faded = context. Click a node for details, *Ask about this* or *Expand neighbours*.
Press `/` to search, `F` to fit, `Esc` to clear. *Replay* walks the evidence path step by step.

**Tabs**: Themes (community cards), Sources (cited chunks), Compare (basic vs auto), Results (benchmark).
"""


def inject() -> None:
    """Add the global stylesheet."""
    st.html(CSS)


def guide() -> None:
    """Guide popover with usage tips."""
    with st.popover("Guide", icon=":material/help:", help="How to use ThemeGraph"):
        st.markdown(GUIDE)
