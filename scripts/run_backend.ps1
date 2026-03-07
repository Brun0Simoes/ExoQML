Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtualenv not found. Run .\scripts\setup.ps1 first."
}

Push-Location .\backend
try {
    ..\.venv\Scripts\uvicorn exoqml.main:app --reload --host 127.0.0.1 --port 8000
} finally {
    Pop-Location
}
