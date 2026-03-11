param(
    [int]$Epochs = 40,
    [int]$MaxPoints = 4096,
    [double]$ReserveFreeGb = 40,
    [double]$DiskUtilization = 0.95,
    [int]$MaxStars = 0,
    [int]$BatchSize = 0,
    [int]$NumWorkers = 0,
    [int]$MinTrainSamples = 512,
    [int]$MinValSamples = 32,
    [int]$MinTestSamples = 32,
    [string]$Device = "auto",
    [switch]$DisableViewCache,
    [switch]$DisableHardNegativeMining,
    [int]$HardNegativeStartEpoch = 6,
    [int]$HardNegativeRefreshEpochs = 4,
    [double]$HardNegativeTopFraction = 0.15,
    [double]$HardNegativeMinScore = 0.55,
    [int]$HardNegativeMinCount = 512,
    [int]$HardNegativeMaxCount = 4096,
    [double]$HardNegativeWeight = 2.5,
    [switch]$NoResume,
    [switch]$EnableCompile,
    [switch]$SkipIngestion
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $projectRoot ".venv\\Scripts\\python.exe"
$backendPath = Join-Path $projectRoot "backend"

if (-not (Test-Path $venvPython)) {
    throw "Virtualenv not found. Run .\\scripts\\setup.ps1 first."
}

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
    "--device", "$Device",
    "--hard-negative-start-epoch", "$HardNegativeStartEpoch",
    "--hard-negative-refresh-epochs", "$HardNegativeRefreshEpochs",
    "--hard-negative-top-fraction", "$HardNegativeTopFraction",
    "--hard-negative-min-score", "$HardNegativeMinScore",
    "--hard-negative-min-count", "$HardNegativeMinCount",
    "--hard-negative-max-count", "$HardNegativeMaxCount",
    "--hard-negative-weight", "$HardNegativeWeight"
)

if ($MaxStars -gt 0) { $argsList += @("--max-stars", "$MaxStars") }
if ($BatchSize -gt 0) { $argsList += @("--batch-size", "$BatchSize") }
if ($NumWorkers -gt 0) { $argsList += @("--num-workers", "$NumWorkers") }
if ($DisableViewCache) { $argsList += "--disable-view-cache" }
if ($DisableHardNegativeMining) { $argsList += "--disable-hard-negative-mining" }
if ($NoResume) { $argsList += "--no-resume" }
if ($EnableCompile) { $argsList += "--enable-compile" }
if ($SkipIngestion) { $argsList += "--skip-ingestion" }

$runtimeRoot = Join-Path $backendPath "data\\runtime_cache"
$homePath = Join-Path $runtimeRoot "home"
$tmpPath = Join-Path $runtimeRoot "tmp"
$astropyCachePath = Join-Path $runtimeRoot "astropy_cache"
$astropyConfigPath = Join-Path $runtimeRoot "astropy_config"
$lightkurveCachePath = Join-Path $runtimeRoot "lightkurve_cache"
$mplConfigPath = Join-Path $runtimeRoot "mplconfig"
$torchHomePath = Join-Path $runtimeRoot "torch_home"
$poochHomePath = Join-Path $runtimeRoot "pooch"
$joblibTmpPath = Join-Path $runtimeRoot "joblib_tmp"
$pipCachePath = Join-Path $runtimeRoot "pip_cache"

@(
    $runtimeRoot,
    $homePath,
    $tmpPath,
    $astropyCachePath,
    $astropyConfigPath,
    $lightkurveCachePath,
    $mplConfigPath,
    $torchHomePath,
    $poochHomePath,
    $joblibTmpPath,
    $pipCachePath
) | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

$env:TMP = $tmpPath
$env:TEMP = $tmpPath
$env:HOME = $homePath
$env:USERPROFILE = $homePath
Remove-Item Env:XDG_CACHE_HOME -ErrorAction SilentlyContinue
Remove-Item Env:XDG_CONFIG_HOME -ErrorAction SilentlyContinue
$env:ASTROPY_CACHE_DIR = $astropyCachePath
$env:ASTROPY_CONFIG_DIR = $astropyConfigPath
$env:LIGHTKURVE_CACHE_DIR = $lightkurveCachePath
$env:MPLCONFIGDIR = $mplConfigPath
$env:TORCH_HOME = $torchHomePath
$env:POOCH_HOME = $poochHomePath
$env:JOBLIB_TEMP_FOLDER = $joblibTmpPath
$env:PIP_CACHE_DIR = $pipCachePath
$env:PYTHONUNBUFFERED = "1"

Write-Host "Runtime cache directories pinned to: $runtimeRoot"

Write-Host "Installing science dependencies (required for real training)..."
& $venvPython -m pip install -e "$backendPath[science]"

$nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($null -ne $nvidiaSmi) {
    Write-Host "NVIDIA GPU detected. Installing CUDA-enabled PyTorch wheel (cu128)..."
    & $venvPython -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128
}

Push-Location $backendPath
try {
    & $venvPython -u @argsList
} finally {
    Pop-Location
}
