"""Central configuration loaded from .env."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GRAPHRAG_API_KEY: str | None = GEMINI_API_KEY
LLM_MODEL_CHEAP: str = os.getenv("LLM_MODEL_CHEAP", "gemini-3.5-flash-lite")
LLM_MODEL_STRONG: str = os.getenv("LLM_MODEL_STRONG", "gemini-3.5-flash-lite")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL") or GEMINI_OPENAI_BASE_URL
GRAPH_ROOT: Path = (PROJECT_ROOT / os.getenv("GRAPH_ROOT", "myproj")).resolve()
OUTPUT_DIR: Path = GRAPH_ROOT / "output"
INPUT_DIR: Path = GRAPH_ROOT / "input"
CACHE_DB: Path = PROJECT_ROOT / "cache.db"
ABSTAIN_MIN_SCORE: int = int(os.getenv("ABSTAIN_MIN_SCORE", "1"))
ROUTER_MIN_CONF: float = float(os.getenv("ROUTER_MIN_CONF", "0.5"))
COMMUNITY_LEVEL: int = int(os.getenv("COMMUNITY_LEVEL", "2"))
OFFLINE: bool = os.getenv("OFFLINE") == "1"
GRAPH_BACKEND: str = os.getenv("GRAPH_BACKEND", "neo4j").strip().lower()
NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str | None = os.getenv("NEO4J_PASSWORD")
KUZU_PATH: Path = (PROJECT_ROOT / os.getenv("KUZU_PATH", "myproj/kuzu")).resolve()
CHROMA_PATH: Path = (PROJECT_ROOT / os.getenv("CHROMA_PATH", "myproj/chroma")).resolve()

if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
