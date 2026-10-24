"""Demo data: a 40-node Aurora Bay graph, 3 communities and canned Responses (MOCK=1 or no index)."""
from __future__ import annotations

import logging
import re
import time
from typing import Callable

import networkx as nx

from backend.contracts import Claim, Evidence, Highlights, Response, Source

logger = logging.getLogger(__name__)

BANNER = ("**Demo data**: a built-in 40-node sample graph with canned answers (MOCK=1 or no GraphRAG index found). "
          "Unset MOCK and build the index to use your own documents.")

COMMUNITIES = {
    "0": "Clean Energy Transition",
    "1": "Coastal Resilience and Flood Defense",
    "2": "Transit and Urban Mobility",
}

# (title, type, community, description)
ENTITIES: list[tuple[str, str, str, str]] = [
    ("AURORA BAY CITY COUNCIL", "organization", "0", "The elected council that adopts Aurora Bay's climate, energy and transport policies."),
    ("MAYOR ELENA RUIZ", "person", "0", "Mayor of Aurora Bay who champions the Clean Energy Plan 2030 and the carbon neutral target."),
    ("CLEAN ENERGY PLAN 2030", "policy", "0", "City plan to reach 80% renewable electricity by 2030 through wind, solar and storage."),
    ("HARBOR WIND FARM", "project", "0", "A 120 MW offshore wind project that supplies about a third of the city's electricity."),
    ("SOLAR ROOFTOPS PROGRAM", "project", "0", "Program that installed solar panels on 4,000 homes and all public schools."),
    ("BAYGRID UTILITY", "organization", "0", "Municipal utility that operates the grid, the battery storage site and the substations."),
    ("GRID BATTERY STORAGE", "technology", "0", "A 60 MWh battery system that smooths wind output and supports the grid during storms."),
    ("COMMUNITY SOLAR COOPERATIVE", "organization", "0", "Resident-owned cooperative that lets renters buy shares in shared solar arrays."),
    ("CARBON NEUTRAL TARGET", "policy", "0", "Commitment to make city operations carbon neutral by 2035."),
    ("NORTHSHORE SUBSTATION", "location", "0", "Substation that connects the offshore cable to the city grid."),
    ("GREEN BOND ISSUE", "event", "0", "A $250 million bond sale that financed wind, storage and seawall projects."),
    ("DR. PRIYA NAIR", "person", "0", "Energy analyst who evaluated the cost and reliability of the Clean Energy Plan 2030."),
    ("HEAT PUMP REBATE", "policy", "0", "Rebate that helps households replace gas boilers with electric heat pumps."),
    ("OFFSHORE CABLE PROJECT", "project", "0", "Subsea cable that brings Harbor Wind Farm power ashore at Northshore."),
    ("COASTAL RESILIENCE OFFICE", "organization", "1", "City office that plans flood defenses and coordinates climate adaptation."),
    ("SEAWALL UPGRADE", "project", "1", "Raising and reinforcing 3 km of seawall along the Harbor District."),
    ("HARBOR DISTRICT", "location", "1", "Low-lying waterfront district that hosts the port and the wind farm operations base."),
    ("FLOOD RISK MAP", "technology", "1", "Interactive map that shows flood exposure for every block under future sea levels."),
    ("STORM SURGE 2024", "event", "1", "The October 2024 storm surge that flooded 600 homes and closed the port for two days."),
    ("WETLAND RESTORATION", "project", "1", "Restoring 200 hectares of salt marsh to absorb storm surge and store carbon."),
    ("MARCUS OKAFOR", "person", "1", "Director of the Coastal Resilience Office who led the storm surge review."),
    ("CLIMATE ADAPTATION FUND", "policy", "1", "Dedicated fund that pays for seawalls, wetlands and flood warning systems."),
    ("TIDE GAUGE NETWORK", "technology", "1", "Sensors that measure water levels in real time around the bay."),
    ("EMERGENCY ALERT SYSTEM", "technology", "1", "Text and siren alerts triggered by tide gauge readings during storms."),
    ("BAY CONSERVANCY", "organization", "1", "Non-profit that manages wetland restoration and monitors bay ecology."),
    ("LOWLANDS NEIGHBORHOOD", "location", "1", "Residential area with the highest flood risk in the city."),
    ("MANAGED RETREAT POLICY", "policy", "1", "Voluntary buyout policy for homes that flood repeatedly."),
    ("AURORA TRANSIT AUTHORITY", "organization", "2", "Agency that runs buses, light rail and the bike network."),
    ("ELECTRIC BUS FLEET", "project", "2", "Replacement of 180 diesel buses with battery electric buses by 2028."),
    ("BAYLINE LIGHT RAIL", "project", "2", "A 14 km light rail line linking Central Station to the Waterfront Corridor."),
    ("BIKE LANE NETWORK", "project", "2", "A 90 km network of protected bike lanes."),
    ("CONGESTION PRICING", "policy", "2", "Downtown charge whose revenue funds transit and fare equity."),
    ("CENTRAL STATION", "location", "2", "Main transit hub where light rail, buses and the mobility hub meet."),
    ("SOFIA CHEN", "person", "2", "Director of the Aurora Transit Authority who leads bus electrification."),
    ("FARE EQUITY PROGRAM", "policy", "2", "Free and reduced fares for low-income riders, students and seniors."),
    ("CHARGING DEPOT", "location", "2", "Bus depot with 120 chargers powered partly by renewable energy."),
    ("RIDERSHIP SURVEY 2025", "event", "2", "Survey showing transit ridership rose 18% after fare equity and new bus routes."),
    ("MOBILITY HUB PILOT", "project", "2", "Pilot combining bike share, e-scooters and bus stops at Central Station."),
    ("WATERFRONT CORRIDOR", "location", "2", "Transit and cycling corridor along the Harbor District waterfront."),
    ("ANNUAL SCORECARD 2025", "event", "2", "Yearly report that tracks progress on energy, resilience and transit goals."),
]

# (source, target, verb phrase)
RELATIONS: list[tuple[str, str, str]] = [
    ("AURORA BAY CITY COUNCIL", "CLEAN ENERGY PLAN 2030", "adopted"),
    ("MAYOR ELENA RUIZ", "CLEAN ENERGY PLAN 2030", "championed"),
    ("MAYOR ELENA RUIZ", "CARBON NEUTRAL TARGET", "announced"),
    ("CLEAN ENERGY PLAN 2030", "HARBOR WIND FARM", "relies on"),
    ("CLEAN ENERGY PLAN 2030", "SOLAR ROOFTOPS PROGRAM", "includes"),
    ("CLEAN ENERGY PLAN 2030", "HEAT PUMP REBATE", "includes"),
    ("CLEAN ENERGY PLAN 2030", "CARBON NEUTRAL TARGET", "supports"),
    ("HARBOR WIND FARM", "OFFSHORE CABLE PROJECT", "is connected through"),
    ("OFFSHORE CABLE PROJECT", "NORTHSHORE SUBSTATION", "lands at"),
    ("BAYGRID UTILITY", "NORTHSHORE SUBSTATION", "operates"),
    ("BAYGRID UTILITY", "GRID BATTERY STORAGE", "operates"),
    ("GRID BATTERY STORAGE", "HARBOR WIND FARM", "stores power from"),
    ("COMMUNITY SOLAR COOPERATIVE", "SOLAR ROOFTOPS PROGRAM", "partners with"),
    ("GREEN BOND ISSUE", "HARBOR WIND FARM", "financed"),
    ("GREEN BOND ISSUE", "GRID BATTERY STORAGE", "financed"),
    ("DR. PRIYA NAIR", "CLEAN ENERGY PLAN 2030", "evaluated"),
    ("DR. PRIYA NAIR", "GRID BATTERY STORAGE", "recommended expanding"),
    ("BAYGRID UTILITY", "HEAT PUMP REBATE", "administers"),
    ("COASTAL RESILIENCE OFFICE", "SEAWALL UPGRADE", "manages"),
    ("COASTAL RESILIENCE OFFICE", "FLOOD RISK MAP", "published"),
    ("MARCUS OKAFOR", "COASTAL RESILIENCE OFFICE", "directs"),
    ("MARCUS OKAFOR", "STORM SURGE 2024", "led the review of"),
    ("STORM SURGE 2024", "HARBOR DISTRICT", "flooded"),
    ("STORM SURGE 2024", "LOWLANDS NEIGHBORHOOD", "damaged homes in"),
    ("SEAWALL UPGRADE", "HARBOR DISTRICT", "protects"),
    ("WETLAND RESTORATION", "LOWLANDS NEIGHBORHOOD", "shields"),
    ("BAY CONSERVANCY", "WETLAND RESTORATION", "manages"),
    ("CLIMATE ADAPTATION FUND", "SEAWALL UPGRADE", "pays for"),
    ("CLIMATE ADAPTATION FUND", "WETLAND RESTORATION", "pays for"),
    ("TIDE GAUGE NETWORK", "EMERGENCY ALERT SYSTEM", "triggers"),
    ("COASTAL RESILIENCE OFFICE", "TIDE GAUGE NETWORK", "operates"),
    ("FLOOD RISK MAP", "LOWLANDS NEIGHBORHOOD", "rates as highest risk"),
    ("MANAGED RETREAT POLICY", "LOWLANDS NEIGHBORHOOD", "offers buyouts in"),
    ("AURORA BAY CITY COUNCIL", "MANAGED RETREAT POLICY", "approved"),
    ("AURORA TRANSIT AUTHORITY", "ELECTRIC BUS FLEET", "is deploying"),
    ("SOFIA CHEN", "AURORA TRANSIT AUTHORITY", "directs"),
    ("SOFIA CHEN", "ELECTRIC BUS FLEET", "leads"),
    ("AURORA TRANSIT AUTHORITY", "BAYLINE LIGHT RAIL", "operates"),
    ("AURORA TRANSIT AUTHORITY", "BIKE LANE NETWORK", "maintains"),
    ("BAYLINE LIGHT RAIL", "CENTRAL STATION", "starts at"),
    ("BAYLINE LIGHT RAIL", "WATERFRONT CORRIDOR", "runs along"),
    ("ELECTRIC BUS FLEET", "CHARGING DEPOT", "charges at"),
    ("CONGESTION PRICING", "FARE EQUITY PROGRAM", "funds"),
    ("FARE EQUITY PROGRAM", "RIDERSHIP SURVEY 2025", "is credited in"),
    ("MOBILITY HUB PILOT", "CENTRAL STATION", "is located at"),
    ("BIKE LANE NETWORK", "WATERFRONT CORRIDOR", "extends along"),
    ("AURORA BAY CITY COUNCIL", "CONGESTION PRICING", "approved"),
    ("ANNUAL SCORECARD 2025", "RIDERSHIP SURVEY 2025", "cites"),
    # cross-community links
    ("CHARGING DEPOT", "BAYGRID UTILITY", "buys renewable power from"),
    ("ELECTRIC BUS FLEET", "CLEAN ENERGY PLAN 2030", "contributes to"),
    ("GREEN BOND ISSUE", "SEAWALL UPGRADE", "financed"),
    ("HARBOR DISTRICT", "HARBOR WIND FARM", "hosts the operations base of"),
    ("WATERFRONT CORRIDOR", "HARBOR DISTRICT", "crosses"),
    ("ANNUAL SCORECARD 2025", "CLEAN ENERGY PLAN 2030", "tracks"),
    ("ANNUAL SCORECARD 2025", "SEAWALL UPGRADE", "tracks"),
    ("GRID BATTERY STORAGE", "EMERGENCY ALERT SYSTEM", "keeps powered"),
]

DOCS = [
    ("01_clean_energy_plan.txt", "0", 0),
    ("02_grid_and_storage.txt", "0", 1),
    ("03_coastal_resilience_strategy.txt", "1", 0),
    ("04_storm_surge_review.txt", "1", 1),
    ("05_transit_master_plan.txt", "2", 0),
    ("06_annual_scorecard_2025.txt", "x", 0),
]

REPORT_SUMMARIES = {
    "0": ("Aurora Bay is shifting to renewable power through the Clean Energy Plan 2030. "
          "Harbor Wind Farm and grid battery storage form the backbone of the new grid. "
          "Rooftop solar, a community solar cooperative and heat pump rebates spread the benefits to households. "
          "A green bond issue financed the largest projects."),
    "1": ("The city is defending its coast after the 2024 storm surge flooded the Harbor District. "
          "The seawall upgrade and wetland restoration are the main physical defenses. "
          "Tide gauges and an emergency alert system give residents early warning. "
          "A managed retreat policy offers buyouts in the Lowlands Neighborhood."),
    "2": ("Transit is being electrified and made more equitable. "
          "The Aurora Transit Authority is replacing diesel buses with an electric bus fleet. "
          "Congestion pricing funds a fare equity program that lifted ridership by 18%. "
          "Bayline Light Rail and protected bike lanes connect Central Station to the waterfront."),
}


def _sentence(src: str, verb: str, dst: str) -> str:
    """Natural sentence for a relationship."""
    return f"{src.title()} {verb} {dst.title()}."


def _doc_sentences() -> dict[str, list[str]]:
    """Sentences per document: entity descriptions and relationship sentences, split by community."""
    comm_of = {t: c for t, _, c, _ in ENTITIES}
    out: dict[str, list[str]] = {name: [] for name, _, _ in DOCS}
    for title, _, comm, desc in ENTITIES:
        half = 0 if sum(map(ord, title)) % 2 == 0 else 1
        target = next((n for n, c, h in DOCS if c == comm and h == half), None) or next(n for n, c, _ in DOCS if c == comm)
        out[target].append(f"{title.title()}: {desc}")
    for i, (s, t, verb) in enumerate(RELATIONS):
        if comm_of[s] != comm_of[t] or s.startswith("ANNUAL") or t.startswith("ANNUAL"):
            out["06_annual_scorecard_2025.txt"].append(_sentence(s, verb, t))
            continue
        docs = [n for n, c, _ in DOCS if c == comm_of[s]]
        out[docs[i % len(docs)]].append(_sentence(s, verb, t))
    return out


def _chunk(sentences: list[str], size: int = 4) -> list[str]:
    """Group sentences into text units."""
    return [" ".join(sentences[i:i + size]) for i in range(0, len(sentences), size)]


class MockData:
    """All demo tables plus a NetworkX graph shaped like backend.graph.load_graph()."""

    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.text_units: dict[str, tuple[str, str]] = {}
        for d_idx, (name, sentences) in enumerate(_doc_sentences().items()):
            self.documents.append({"id": f"doc_{d_idx}", "title": name, "text": " ".join(sentences)})
            for k, text in enumerate(_chunk(sentences)):
                self.text_units[f"tu_{d_idx}_{k}"] = (name, text)
        self.titles = dict(COMMUNITIES)
        self.graph = self._build_graph()
        self.reports = [{"community": c, "title": t, "summary": REPORT_SUMMARIES[c], "content": REPORT_SUMMARIES[c], "level": 0,
                         "rank": 9.0 - int(c) * 0.7, "size": sum(1 for e in ENTITIES if e[2] == c)}
                        for c, t in COMMUNITIES.items()]

    def _units_mentioning(self, *names: str) -> list[str]:
        """Text unit ids whose text mentions every given name."""
        return [uid for uid, (_, text) in self.text_units.items()
                if all(n.title() in text for n in names)]

    def _build_graph(self) -> nx.Graph:
        g = nx.Graph()
        for title, etype, comm, desc in ENTITIES:
            g.add_node(title, entity_id=f"e_{len(g)}", type=etype, description=desc, community=comm,
                       communities=[comm], text_unit_ids=self._units_mentioning(title))
        for i, (s, t, verb) in enumerate(RELATIONS):
            g.add_edge(s, t, id=str(i), uuid=f"r_{i}", description=_sentence(s, verb, t), weight=1.0,
                       text_unit_ids=self._units_mentioning(s, t))
        for n in g.nodes:
            g.nodes[n]["degree"] = g.degree(n)
        return g

    def unit_for(self, sentence: str) -> tuple[str, str, str]:
        """(unit id, doc, text) of the text unit containing a sentence."""
        for uid, (doc, text) in self.text_units.items():
            if sentence in text:
                return uid, doc, text
        raise KeyError(sentence)

    def edge_ids(self, nodes: list[str]) -> list[str]:
        """Relationship ids among a node set."""
        return [d["id"] for _, _, d in self.graph.subgraph(nodes).edges(data=True)]


def _claim_from_relation(data: MockData, rel_index: int, text: str) -> tuple[Claim, Source]:
    """A claim whose quote is the relationship sentence, with its source chunk."""
    s, t, verb = RELATIONS[rel_index]
    quote = _sentence(s, verb, t)
    uid, doc, unit = data.unit_for(quote)
    return Claim(text=text, evidence=[Evidence(chunk_id=uid, quote=quote)]), Source(doc=doc, chunk_id=uid, text=unit)


def _rel(s: str, t: str) -> int:
    """Index of a relationship by its endpoints."""
    return next(i for i, (a, b, _) in enumerate(RELATIONS) if {a, b} == {s, t})


def _global_response(data: MockData) -> Response:
    claims = []
    for cid, summary in REPORT_SUMMARIES.items():
        first, second = re.split(r"(?<=\.)\s+", summary)[:2]
        claims.append(Claim(text=f"{COMMUNITIES[cid]}: {first}", evidence=[
            Evidence(chunk_id=f"report_{cid}", quote=first), Evidence(chunk_id=f"report_{cid}", quote=second)]))
    nodes = [n for c in COMMUNITIES for n in sorted(
        (n for n, d in data.graph.nodes(data=True) if d["community"] == c),
        key=lambda n: -data.graph.nodes[n]["degree"])[:4]]
    answer = ("The documents describe **three connected themes**:\n\n"
              "1. **Clean Energy Transition**: wind, storage and rooftop solar under the Clean Energy Plan 2030.\n"
              "2. **Coastal Resilience and Flood Defense**: seawalls, wetlands and early warning after the 2024 storm surge.\n"
              "3. **Transit and Urban Mobility**: electric buses, light rail and fare equity funded by congestion pricing.\n\n"
              "The themes meet in shared financing (the green bond) and in the Harbor District.")
    return Response(route="global", answer=answer, claims=claims,
                    highlights=Highlights(nodes=nodes, edges=data.edge_ids(nodes), communities=list(COMMUNITIES)),
                    router={"route": "global", "confidence": 0.93, "reason": "Asks for overarching themes across the corpus."},
                    timings_ms={"route": 410, "search": 2350, "answer": 1180, "verify": 640, "path": 35, "total": 4615},
                    stats={"chunks": 3, "quotes_total": 6, "quotes_valid": 6, "claims_raw": 3, "claims_verified": 3},
                    search_answer="Three themes: clean energy, coastal resilience, and transit.")


def _local_response(data: MockData) -> Response:
    pairs = [(_rel("CLEAN ENERGY PLAN 2030", "HARBOR WIND FARM"), "Harbor Wind Farm is a core part of the Clean Energy Plan 2030."),
             (_rel("HARBOR WIND FARM", "OFFSHORE CABLE PROJECT"), "Its power reaches the city through the offshore cable."),
             (_rel("GREEN BOND ISSUE", "HARBOR WIND FARM"), "The project was financed by the green bond issue.")]
    made = [_claim_from_relation(data, i, text) for i, text in pairs]
    nodes = ["HARBOR WIND FARM", "CLEAN ENERGY PLAN 2030", "OFFSHORE CABLE PROJECT", "GREEN BOND ISSUE", "GRID BATTERY STORAGE"]
    answer = ("**Harbor Wind Farm** is a 120 MW offshore wind project at the heart of the Clean Energy Plan 2030. "
              "Its output comes ashore through the offshore cable at Northshore, is smoothed by grid battery storage, "
              "and the build was financed by the city's green bond issue.")
    return Response(route="local", answer=answer, claims=[c for c, _ in made], sources=[s for _, s in made],
                    highlights=Highlights(nodes=nodes, edges=data.edge_ids(nodes), communities=["0"]),
                    router={"route": "local", "confidence": 0.95, "reason": "Asks about one named entity."},
                    timings_ms={"route": 380, "search": 1900, "answer": 950, "verify": 520, "path": 20, "total": 3770},
                    stats={"chunks": 6, "quotes_total": 3, "quotes_valid": 3, "claims_raw": 3, "claims_verified": 3},
                    search_answer="Harbor Wind Farm is a 120 MW offshore wind farm.")


def _path_response(data: MockData) -> Response:
    g = data.graph
    core = nx.shortest_path(g, "CLEAN ENERGY PLAN 2030", "ELECTRIC BUS FLEET")
    core += [n for n in nx.shortest_path(g, "ELECTRIC BUS FLEET", "SEAWALL UPGRADE") if n not in core]
    extra = [n for n in ["BAYGRID UTILITY", "CHARGING DEPOT", "GREEN BOND ISSUE", "SOFIA CHEN", "AURORA TRANSIT AUTHORITY",
                         "HARBOR WIND FARM", "ANNUAL SCORECARD 2025", "CLIMATE ADAPTATION FUND", "GRID BATTERY STORAGE",
                         "HARBOR DISTRICT", "MAYOR ELENA RUIZ"] if n not in core]
    nodes = (core + extra)[:12]
    pairs = [(_rel("ELECTRIC BUS FLEET", "CHARGING DEPOT"), "Electric buses charge at the Charging Depot."),
             (_rel("CHARGING DEPOT", "BAYGRID UTILITY"), "The depot buys renewable power from BayGrid Utility."),
             (_rel("ELECTRIC BUS FLEET", "CLEAN ENERGY PLAN 2030"), "Bus electrification counts toward the Clean Energy Plan 2030."),
             (_rel("GREEN BOND ISSUE", "SEAWALL UPGRADE"), "The same green bond that financed clean energy also paid for the seawall upgrade.")]
    made = [_claim_from_relation(data, i, text) for i, text in pairs]
    answer = ("The energy and transit themes are linked through **power supply and financing**. "
              "Electric buses charge at a depot that buys renewable power from BayGrid Utility, so bus electrification "
              "advances the Clean Energy Plan 2030. The green bond ties both to coastal defense by also funding the seawall upgrade.")
    return Response(route="local", answer=answer, claims=[c for c, _ in made], sources=[s for _, s in made],
                    highlights=Highlights(nodes=nodes, edges=data.edge_ids(nodes), communities=["0", "1", "2"]),
                    router={"route": "local", "confidence": 0.81, "reason": "Asks how specific entities are connected (multi-hop)."},
                    timings_ms={"route": 420, "search": 2600, "answer": 1300, "verify": 700, "path": 60, "total": 5080},
                    stats={"chunks": 8, "quotes_total": 4, "quotes_valid": 4, "claims_raw": 4, "claims_verified": 4},
                    search_answer="Buses use renewable power; the green bond funds both energy and seawalls.")


def _abstain_response() -> Response:
    reason = ("This question looks outside the scope of the indexed documents. "
              f"The closest topic in the corpus is \u201c{COMMUNITIES['2']}\u201d.")
    return Response(route="abstain", answer="I can't answer that from the indexed documents.", abstained=True,
                    abstain_reason=reason,
                    router={"route": "abstain", "confidence": 0.9, "reason": "Unrelated to the corpus (general knowledge)."},
                    timings_ms={"route": 350, "total": 350})


def _basic_response(data: MockData, base: Response) -> Response:
    """Baseline vector-search variant: plain chunks, no graph, fewer verified claims."""
    claims = [c for c in base.claims if not c.evidence[0].chunk_id.startswith("report_")][:1]
    if not claims:
        claim, _ = _claim_from_relation(data, _rel("CLEAN ENERGY PLAN 2030", "HARBOR WIND FARM"),
                                        "The Clean Energy Plan 2030 relies on Harbor Wind Farm.")
        claims = [claim]
    sources = [Source(doc=data.text_units[c.evidence[0].chunk_id][0], chunk_id=c.evidence[0].chunk_id,
                      text=data.text_units[c.evidence[0].chunk_id][1]) for c in claims]
    answer = ("Based on the closest matching passages, the documents mention the Clean Energy Plan 2030 and "
              "Harbor Wind Farm. The retrieved chunks do not cover the other topics, so this answer is partial.")
    return Response(route="basic", answer=answer, claims=claims, sources=sources,
                    router={"route": "basic", "confidence": 1.0, "reason": "Forced mode: basic"},
                    timings_ms={"route": 0, "search": 900, "answer": 800, "verify": 400, "path": 5, "total": 2105},
                    stats={"chunks": 5, "quotes_total": 2, "quotes_valid": 1, "claims_raw": 2, "claims_verified": 1},
                    search_answer="Partial answer from top-k chunks.")


OUT_OF_SCOPE = ("capital", "world cup", "pizza", "weather", "stock", "recipe", "football", "president of")
PATH_WORDS = ("connect", "link", "path", "relationship between", "tie")
THEME_WORDS = ("theme", "main", "overall", "across", "summar", "trend", "big picture", "relate")


def pick(question: str, mode: str, data: MockData) -> Response:
    """Choose a canned Response for a question and mode."""
    q = question.lower()
    if mode == "basic":
        return _basic_response(data, pick(question, "auto", data))
    if mode == "global":
        return _global_response(data)
    if mode == "auto" and any(w in q for w in OUT_OF_SCOPE):
        return _abstain_response()
    if any(w in q for w in PATH_WORDS):
        return _path_response(data)
    if mode == "auto" and any(w in q for w in THEME_WORDS):
        return _global_response(data)
    return _local_response(data)


def run(question: str, mode: str = "auto", progress: Callable[[str], None] | None = None,
        data: MockData | None = None, delay_s: float = 0.35) -> Response:
    """Mock pipeline.run: same signature shape, canned answers, short simulated stages."""
    data = data or MockData()
    for stage in ("routing", "searching", "verifying"):
        if progress:
            progress(stage)
        time.sleep(delay_s)
    resp = pick(question, mode, data)
    if mode in ("global", "local") and resp.route != mode:
        resp = resp.model_copy(update={"route": mode})
    if mode != "auto":
        resp = resp.model_copy(update={"router": {"route": mode, "confidence": 1.0, "reason": f"Forced mode: {mode}"}})
    logger.info("Mock answer for %r -> %s", question, resp.route)
    return resp
