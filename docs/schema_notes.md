# GraphRAG output schema notes

Provisional: taken from the installed package (`graphrag/data_model/schemas.py`, graphrag 3.3.0).
`scripts/inspect_output.py` overwrites this file with real row counts and dtypes after indexing.

## Link checks (to confirm after indexing)

- relationships carry `text_unit_ids` (lets `path.py` turn edges into source quotes)
- text_units link to documents via `document_id`

## Tables (`myproj/output/*.parquet`)

| table | columns |
|---|---|
| entities | `id`, `human_readable_id`, `title`, `type`, `description`, `text_unit_ids`, `frequency`, `degree` |
| relationships | `id`, `human_readable_id`, `source`, `target`, `description`, `weight`, `combined_degree`, `text_unit_ids` |
| communities | `id`, `human_readable_id`, `community`, `level`, `parent`, `children`, `title`, `entity_ids`, `relationship_ids`, `text_unit_ids`, `period`, `size` |
| community_reports | `id`, `human_readable_id`, `community`, `level`, `parent`, `children`, `title`, `summary`, `full_content`, `rank`, `rating_explanation`, `findings`, `full_content_json`, `period`, `size` |
| text_units | `id`, `human_readable_id`, `text`, `n_tokens`, `document_id`, `entity_ids`, `relationship_ids`, `covariate_ids` |
| documents | `id`, `human_readable_id`, `title`, `text`, `text_unit_ids`, `creation_date`, `raw_data` |

## Query context id conventions (graphrag 3.3)

- global `reports[].id` = community number (`community_reports.community`)
- local `entities[].entity` = `entities.title`; `entities[].id` = `entities.human_readable_id`
- local `relationships[].id` = `relationships.human_readable_id`
- `sources[].id` = row position of the text unit; ThemeGraph maps sources to documents by text match
- entities have no community column; membership comes from `communities.entity_ids` (per level)
