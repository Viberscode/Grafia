# ThemeGraph 3-minute demo script

**Before recording:**
- Run `.\tasks.ps1 check` (all PASS).
- Start the app with `.\tasks.ps1 app` in a wide browser window, mode Auto, chat empty.
- Ask every question below once beforehand, so the answers come from `cache.db` instantly.
- Keep a spare text file to upload (any short `.txt`).

## 0:00-0:25 Upload and build
- "ThemeGraph turns documents into a knowledge graph in a graph database and a vector database, and answers with
  verified quotes."
- Open **Documents**, drop the `.txt` file, press **Upload**, then **Build knowledge graph**.
- Point at the estimate (chunks, LLM calls), the live stages and the header chips: **Kùzu/Neo4j** and **ChromaDB** counts.
- Building takes minutes on the free tier, so say "this runs in the background" and continue while it runs.

## 0:25-0:55 Thematic question (global, community halo)
- Ask: **"What are the main themes of Aurora Bay's clean energy transition?"**
- Point at the **Global** badge and the **3-step trace** badge; each claim carries a **theme** chip.
- Graph: the community halo surrounds the theme's members, and only entities named in the reports stay red.

## 0:55-1:45 Multi-hop question (numbered path, click a document)
- Ask: **"How is Dr. Priya Raman connected to the company that built the Kestrel control software?"**
- **Local** badge, **4-step trace**. Press **Replay**: the captions walk through seed, hop, chunk and document.
- The numbered badges mark the path DR. PRIYA RAMAN → … → GRIDWISE ANALYTICS.
- Click a square document node (`07_coastal_state_research.txt`): the drawer shows the excerpts, the trace steps
  that used it, and *Ask about this*.

## 1:45-2:10 Hybrid question
- Ask: **"What are the main themes around financing, and who issued the green bond?"**
- **Hybrid** badge with the **Theme N + Facts M** split; claims show **theme** or **fact** origin chips.

## 2:10-2:25 Unanswerable question (abstain)
- Ask: **"What was Mayor Elena Vasquez's salary in 2023?"**
- The system declines and names the closest theme instead of guessing.

## 2:25-2:45 Compare tab
- Open **Compare**, then **Run GraphRAG basic for this question** on the multi-hop question.
- ThemeGraph (with its trace) sits next to GraphRAG basic; words and documents unique to one side are highlighted.

## 2:45-3:00 Results tab
- Open **Results**: per-system bars for correctness, evidence recall, quote validity and router accuracy, plus
  latency and LLM calls, with a question-type filter and extraction precision once the audit is scored.
- "Every answer ships with verifiable evidence and an exact retrieval path, and the benchmark is reproducible
  with `.\tasks.ps1 eval`."

## Fallbacks
- **Out of quota:** `.\tasks.ps1 demo` (or `streamlit run app.py -- --demo-seed`) preloads the thematic,
  multi-hop and hybrid answers from `docs/demo_seed.json`; the graph, Replay, drawer and Results tab all work offline.
  Re-record with `python -m ui.demo_seed --record`.
- **No index or no key:** `.\tasks.ps1 mock` runs the full UI on built-in sample data (`MOCK=1`).
