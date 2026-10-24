# Windows equivalent of the Makefile: .\tasks.ps1 up|load|index|inspect|graph|app|mock|demo|eval|smoke|test|check
param([Parameter(Mandatory)][ValidateSet("up", "load", "index", "inspect", "graph", "app", "mock", "demo", "eval", "smoke", "test", "check")][string]$Task)
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Set-Location $PSScriptRoot

function Invoke-Check([string]$Name, [scriptblock]$Body) {
    & $Body
    $ok = $LASTEXITCODE -eq 0
    $script:checks += [pscustomobject]@{ Check = $Name; Result = $(if ($ok) { "PASS" } else { "FAIL" }) }
}

switch ($Task) {
    "up"      { docker compose up -d --wait neo4j }
    "load"    { & $py -m backend.stores.loader --reset }
    "index"   { & $py -m backend.engine --index; if ($LASTEXITCODE -eq 0) { & $py scripts/inspect_output.py } }
    "inspect" { & $py scripts/inspect_output.py }
    "graph"   { & $py -m backend.graph }
    "app"     { & $py -m streamlit run app.py }
    "mock"    { $env:MOCK = "1"; & $py -m streamlit run app.py; Remove-Item Env:MOCK }
    "demo"    { & $py -m streamlit run app.py -- --demo-seed }
    "eval"    { & $py -m eval.run --resume --sleep 4; & $py -m eval.report }
    "smoke"   { & $py scripts/smoke_test.py }
    "test"    { $env:OFFLINE = "1"; & $py -m pytest -q tests; Remove-Item Env:OFFLINE }
    "check"   {
        $script:checks = @()
        Invoke-Check "offline tests" { $env:OFFLINE = "1"; & $py -m pytest -q tests; Remove-Item Env:OFFLINE }
        Invoke-Check "store health" { & $py -m backend.stores.health }
        Invoke-Check "smoke test (3 questions)" { & $py scripts/smoke_test.py }
        Write-Host ""
        foreach ($c in $script:checks) {
            Write-Host ("{0}  {1}" -f $c.Result, $c.Check) -ForegroundColor $(if ($c.Result -eq "PASS") { "Green" } else { "Red" })
        }
        $global:LASTEXITCODE = [int](@($script:checks | Where-Object Result -eq "FAIL").Count -gt 0)
    }
}
exit $LASTEXITCODE
