# Windows equivalent of the Makefile: .\tasks.ps1 up|load|index|inspect|graph|app|mock|eval|smoke|test
param([Parameter(Mandatory)][ValidateSet("up", "load", "index", "inspect", "graph", "app", "mock", "eval", "smoke", "test")][string]$Task)
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Set-Location $PSScriptRoot
switch ($Task) {
    "up"      { docker compose up -d --wait neo4j }
    "load"    { & $py -m backend.stores.loader --reset }
    "index"   { & $py -m backend.engine --index; if ($LASTEXITCODE -eq 0) { & $py scripts/inspect_output.py } }
    "inspect" { & $py scripts/inspect_output.py }
    "graph"   { & $py -m backend.graph }
    "app"     { & $py -m streamlit run app.py }
    "mock"    { $env:MOCK = "1"; & $py -m streamlit run app.py; Remove-Item Env:MOCK }
    "eval"    { & $py -m eval.run; & $py -m eval.report }
    "smoke"   { & $py scripts/smoke_test.py }
    "test"    { $env:OFFLINE = "1"; & $py -m pytest -q tests; Remove-Item Env:OFFLINE }
}
exit $LASTEXITCODE
