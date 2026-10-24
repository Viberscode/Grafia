PY ?= python

.PHONY: index inspect app mock eval smoke test graph

index:
	$(PY) -m backend.engine --index
	$(PY) scripts/inspect_output.py

inspect:
	$(PY) scripts/inspect_output.py

graph:
	$(PY) -m backend.graph

app:
	streamlit run app.py

mock:
	MOCK=1 streamlit run app.py

eval:
	$(PY) -m eval.run --resume --sleep 4
	$(PY) -m eval.report

smoke:
	$(PY) scripts/smoke_test.py

test:
	OFFLINE=1 $(PY) -m pytest -q tests
