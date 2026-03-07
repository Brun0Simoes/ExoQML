param(
    [int]$Epochs = 40,
    [int]$MaxPoints = 4096,
    [double]$ReserveFreeGb = 60,
    [double]$DiskUtilization = 0.90,
    [int]$MaxStars = 0,
    [int]$BatchSize = 0,
    [int]$NumWorkers = 0,
    [int]$MinTrainSamples = 512,
    [int]$MinValSamples = 32,
    [int]$MinTestSamples = 32,
    [string]$Device = "auto",
    [switch]$NoResume,
    [switch]$EnableCompile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtualenv not found. Run .\scripts\setup.ps1 first."
}

Write-Host "Installing science dependencies (required for real training)..."
.\.venv\Scripts\python -m pip install -e .\backend[science]

$argsList = @(
    "-m", "exoqml.training.max_train",
    "--dataset-root", "./data/train_max",
    "--epochs", "$Epochs",
    "--max-points", "$MaxPoints",
    "--reserve-free-gb", "$ReserveFreeGb",
    "--disk-utilization", "$DiskUtilization",
    "--min-train-samples", "$MinTrainSamples",
    "--min-val-samples", "$MinValSamples",
    "--min-test-samples", "$MinTestSamples",
    "--device", "$Device"
)

if ($MaxStars -gt 0) { $argsList += @("--max-stars", "$MaxStars") }
if ($BatchSize -gt 0) { $argsList += @("--batch-size", "$BatchSize") }
if ($NumWorkers -gt 0) { $argsList += @("--num-workers", "$NumWorkers") }
if ($NoResume) { $argsList += "--no-resume" }
if ($EnableCompile) { $argsList += "--enable-compile" }

Push-Location .\backend
try {
    ..\.venv\Scripts\python @argsList
} finally {
    Pop-Location
}
