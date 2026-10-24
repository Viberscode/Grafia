"""Central configuration loaded from .env."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GRAPHRAG_API_KEY: str | None = os.getenv("GRAPHRAG_API_KEY")
LLM_MODEL_CHEAP: str = os.getenv("LLM_MODEL_CHEAP", "gpt-4o-mini")
LLM_MODEL_STRONG: str = os.getenv("LLM_MODEL_STRONG", "gpt-4o")
LLM_BASE_URL: str | None = os.getenv("LLM_BASE_URL") or None
GRAPH_ROOT: Path = (PROJECT_ROOT / os.getenv("GRAPH_ROOT", "myproj")).resolve()
OUTPUT_DIR: Path = GRAPH_ROOT / "output"
INPUT_DIR: Path = GRAPH_ROOT / "input"
CACHE_DB: Path = PROJECT_ROOT / "cache.db"
ABSTAIN_MIN_SCORE: int = int(os.getenv("ABSTAIN_MIN_SCORE", "1"))
ROUTER_MIN_CONF: float = float(os.getenv("ROUTER_MIN_CONF", "0.5"))
COMMUNITY_LEVEL: int = int(os.getenv("COMMUNITY_LEVEL", "2"))
OFFLINE: bool = os.getenv("OFFLINE") == "1"

if GRAPHRAG_API_KEY:
    os.environ["GRAPHRAG_API_KEY"] = GRAPHRAG_API_KEY

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
