param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("export", "import")]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [string]$Path,

    [string]$CodexRoot = "$HOME\.codex"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Directory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DirectoryPath
    )

    if (-not (Test-Path -LiteralPath $DirectoryPath)) {
        New-Item -ItemType Directory -Path $DirectoryPath | Out-Null
    }
}

function Read-JsonLines {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath
    )

    if (-not (Test-Path -LiteralPath $FilePath)) {
        return @()
    }

    $items = New-Object System.Collections.Generic.List[object]
    foreach ($line in Get-Content -LiteralPath $FilePath) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        $items.Add(($line | ConvertFrom-Json))
    }

    return $items
}

function Merge-SessionIndex {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceIndexPath,

        [Parameter(Mandatory = $true)]
        [string]$TargetIndexPath
    )

    $mergedById = @{}

    foreach ($entry in Read-JsonLines -FilePath $TargetIndexPath) {
        $mergedById[$entry.id] = $entry
    }

    foreach ($entry in Read-JsonLines -FilePath $SourceIndexPath) {
        $mergedById[$entry.id] = $entry
    }

    $orderedEntries = $mergedById.Values | Sort-Object {
        if ($_.updated_at) {
            [datetime]$_.updated_at
        } else {
            [datetime]::MinValue
        }
    }

    $lines = foreach ($entry in $orderedEntries) {
        $entry | ConvertTo-Json -Compress
    }

    if ($lines.Count -gt 0) {
        Set-Content -LiteralPath $TargetIndexPath -Value $lines
    }
}

function Copy-SessionFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceSessionsRoot,

        [Parameter(Mandatory = $true)]
        [string]$TargetSessionsRoot
    )

    if (-not (Test-Path -LiteralPath $SourceSessionsRoot)) {
        throw "Source sessions folder not found: $SourceSessionsRoot"
    }

    Ensure-Directory -DirectoryPath $TargetSessionsRoot

    $sourceRootResolved = (Resolve-Path -LiteralPath $SourceSessionsRoot).Path
    $sourceFiles = Get-ChildItem -LiteralPath $sourceRootResolved -Recurse -File

    foreach ($file in $sourceFiles) {
        $relativePath = $file.FullName.Substring($sourceRootResolved.Length).TrimStart('\')
        $destinationPath = Join-Path $TargetSessionsRoot $relativePath
        $destinationDirectory = Split-Path -Parent $destinationPath
        Ensure-Directory -DirectoryPath $destinationDirectory
        Copy-Item -LiteralPath $file.FullName -Destination $destinationPath -Force
    }
}

$codexRootResolved = [System.IO.Path]::GetFullPath($CodexRoot)
$transferRootResolved = [System.IO.Path]::GetFullPath($Path)
$sessionsPath = Join-Path $codexRootResolved "sessions"
$sessionIndexPath = Join-Path $codexRootResolved "session_index.jsonl"

if ($Mode -eq "export") {
    if (-not (Test-Path -LiteralPath $sessionsPath)) {
        throw "Codex sessions folder not found: $sessionsPath"
    }

    Ensure-Directory -DirectoryPath $transferRootResolved
    $exportSessionsPath = Join-Path $transferRootResolved "sessions"
    Ensure-Directory -DirectoryPath $exportSessionsPath

    Copy-SessionFiles -SourceSessionsRoot $sessionsPath -TargetSessionsRoot $exportSessionsPath

    if (Test-Path -LiteralPath $sessionIndexPath) {
        Copy-Item -LiteralPath $sessionIndexPath -Destination (Join-Path $transferRootResolved "session_index.jsonl") -Force
    }

    $manifest = [ordered]@{
        exported_at = (Get-Date).ToString("o")
        source_codex_root = $codexRootResolved
        includes = @("sessions", "session_index.jsonl")
        excludes = @("auth.json", "state_*.sqlite", "logs_*.sqlite", "memories")
    } | ConvertTo-Json

    Set-Content -LiteralPath (Join-Path $transferRootResolved "manifest.json") -Value $manifest
    Write-Host "Exported Codex history to $transferRootResolved"
    exit 0
}

$importSessionsPath = Join-Path $transferRootResolved "sessions"
$importIndexPath = Join-Path $transferRootResolved "session_index.jsonl"

Ensure-Directory -DirectoryPath $codexRootResolved
Ensure-Directory -DirectoryPath $sessionsPath

Copy-SessionFiles -SourceSessionsRoot $importSessionsPath -TargetSessionsRoot $sessionsPath
Merge-SessionIndex -SourceIndexPath $importIndexPath -TargetIndexPath $sessionIndexPath

Write-Host "Imported Codex history into $codexRootResolved"
