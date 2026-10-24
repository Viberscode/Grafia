# Recommended GraphRAG settings (`myproj/settings.yaml`)

These values are already applied in `myproj/settings.yaml` (graphrag 3.3.0).

| Setting | Value | Why |
|---|---|---|
| `completion_models.default_completion_model.model` | `gpt-4o-mini` | Cheap extraction and community reports |
| `embedding_models.default_embedding_model.model` | `text-embedding-3-small` | Small, cheap embeddings for local/basic search |
| `*.retry.max_retries` | `5` | Fail fast on a dead key instead of retrying forever |
| `chunking.size` / `chunking.overlap` | `500` / `50` | Small chunks give precise citations |
| `extract_graph.max_gleanings` | `1` | One extra extraction pass; more costs money for little gain |
| `extract_graph.entity_types` | `organization, person, location, project, technology, policy, event` | Fixed list for the energy-transition corpus |
| `community_reports.max_length` | `1000` | Short reports keep global search cheap |
| `extract_claims.enabled` | `false` | Covariates are not used by ThemeGraph |

## Notes

- GraphRAG resolves prompt paths and the cache folder relative to the current working directory.
  `backend/engine.py` runs every GraphRAG call with cwd set to `myproj/` and passes absolute
  `cache.storage.base_dir` and `vector_store.db_uri` overrides.
- The API key is read from the environment (`GRAPHRAG_API_KEY` in the root `.env`, loaded by
  `backend/config.py`). Index with `python -m backend.engine --index` so the key is picked up.
  If you use the `graphrag index` CLI directly, put the key in `myproj/.env` first.
- Indexing this 10-document corpus is roughly 60-100 LLM calls with gpt-4o-mini (a few cents).
