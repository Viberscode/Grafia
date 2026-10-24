# ThemeGraph 3-minute demo script

**Setup before recording:** index built, `streamlit run app.py` open in a wide browser window, mode = Auto,
chat cleared. Run each question once beforehand so answers come from `cache.db` instantly.

## 0:00-0:30 Graph overview
- "ThemeGraph turns a folder of documents into a knowledge graph and answers questions with verified quotes."
- Point at the sidebar: communities GraphRAG detected (grid resilience, offshore wind, financing...).
- Tick **Show full graph**: colours are communities. Untick it.

## 0:30-1:10 Thematic question (global route)
- Ask: **"What are the main themes of Aurora Bay's clean energy transition?"**
- Point at the purple **GLOBAL** badge and router confidence.
- Graph: the communities used light up in red.
- Open **Evidence**: every claim has a quote copied from a community report.

## 1:10-2:00 Multi-hop question (local route, path highlight)
- Ask: **"How is Dr. Priya Raman connected to the company that built the Kestrel control software?"**
- Blue **LOCAL** badge. Graph shows the path PRIYA RAMAN, GRIDWISE ANALYTICS, KESTREL BATTERY STORAGE PROJECT in red.
- Open **Sources**: the excerpts come from `07_coastal_state_research` and `04_kestrel_battery_storage`.
- In **Inspect a node**, pick GRIDWISE ANALYTICS to show its description and source text.

## 2:00-2:25 Unanswerable question (abstain)
- Ask: **"What was Mayor Elena Vasquez's salary in 2023?"**
- The system declines and names the closest topic instead of guessing.

## 2:25-3:00 Results
- Show the results table in `README.md` (from `make eval`).
- "Compared with GraphRAG basic, local-only and global-only search, routing gets the right search type
  per question, and the quote check means no answer ships without verifiable evidence."
