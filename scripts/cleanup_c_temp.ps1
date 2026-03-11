param(
    [switch]$IncludeAstropy
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-DirSizeBytes([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    $files = Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue
    if ($null -eq $files) {
        return 0
    }

    $measure = $files | Measure-Object -Property Length -Sum
    $sumProp = $measure.PSObject.Properties["Sum"]
    if ($null -eq $sumProp -or $null -eq $sumProp.Value) {
        return 0
    }

    return [int64]$sumProp.Value
}

$targets = @(
    "$env:LOCALAPPDATA\\Temp",
    "$env:LOCALAPPDATA\\pip\\Cache",
    "$env:USERPROFILE\\.cache\\pip"
)

if ($IncludeAstropy) {
    $targets += "$env:USERPROFILE\\.astropy"
}

$rows = @()
foreach ($target in $targets) {
    if (-not (Test-Path -LiteralPath $target)) {
        $rows += [PSCustomObject]@{
            Path     = $target
            BeforeGB = 0
            AfterGB  = 0
            FreedGB  = 0
            Removed  = 0
            Failed   = 0
            Status   = "not_found"
        }
        continue
    }

    $before = Get-DirSizeBytes -Path $target
    $removed = 0
    $failed = 0

    Get-ChildItem -LiteralPath $target -Force -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop
            $removed++
        }
        catch {
            $failed++
        }
    }

    $after = Get-DirSizeBytes -Path $target
    $rows += [PSCustomObject]@{
        Path     = $target
        BeforeGB = [math]::Round(($before / 1GB), 3)
        AfterGB  = [math]::Round(($after / 1GB), 3)
        FreedGB  = [math]::Round((($before - $after) / 1GB), 3)
        Removed  = $removed
        Failed   = $failed
        Status   = "cleaned"
    }
}

$rows | Format-Table -AutoSize
$totalFreed = [math]::Round((($rows | Measure-Object -Property FreedGB -Sum).Sum), 3)
Write-Output "TOTAL_FREED_GB=$totalFreed"
