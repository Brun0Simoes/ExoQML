Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "[1/4] Creating Python virtualenv (.venv) with Python 3.11 via uv..."
uv venv --python 3.11 .venv

Write-Host "[2/4] Installing backend base dependencies..."
.\.venv\Scripts\python -m ensurepip --upgrade
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e .\backend[dev]

Write-Host "[3/4] Installing optional science stack (lightkurve + torch) ..."
Write-Host "      If this fails, backend will still run with resilient fallback mode."
try {
    .\.venv\Scripts\python -m pip install -e .\backend[science]
} catch {
    Write-Warning "Science stack installation failed. Continuing with base backend."
}

Write-Host "[4/4] Installing frontend dependencies..."
npm.cmd install --prefix .\frontend

Write-Host "Setup finished."
