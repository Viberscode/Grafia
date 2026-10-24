PY ?= python

.PHONY: index inspect app eval smoke test graph

index:
	$(PY) -m backend.engine --index
	$(PY) scripts/inspect_output.py

inspect:
	$(PY) scripts/inspect_output.py

graph:
	$(PY) -m backend.graph

app:
	streamlit run app.py

eval:
	$(PY) -m eval.run
	$(PY) -m eval.report

smoke:
	$(PY) scripts/smoke_test.py

test:
	OFFLINE=1 $(PY) -m pytest -q tests
