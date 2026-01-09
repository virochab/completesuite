# PowerShell script to run Garak tests against the RAG API endpoint
# This uses the .venv-garak environment to avoid dependency conflicts

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ParentDir = Split-Path -Parent $ProjectRoot
$GarakVenv = Join-Path $ParentDir ".venv-garak"
$ConfigFile = Join-Path $ProjectRoot "config\garak_rag_config.json"
$ApiUrl = if ($env:RAG_API_URL) { $env:RAG_API_URL } else { "http://localhost:8000" }

if (-not (Test-Path $GarakVenv)) {
    Write-Host "Error: .venv-garak not found at $GarakVenv" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $ConfigFile)) {
    Write-Host "Error: Garak config file not found at $ConfigFile" -ForegroundColor Red
    exit 1
}

$GarakExe = Join-Path $GarakVenv "Scripts\garak.exe"

if (-not (Test-Path $GarakExe)) {
    Write-Host "Error: Garak executable not found at $GarakExe" -ForegroundColor Red
    exit 1
}

# Update URI in config if RAG_API_URL is set
if ($env:RAG_API_URL) {
    $ConfigContent = Get-Content $ConfigFile -Raw
    $ConfigContent = $ConfigContent -replace "http://localhost:8000", $ApiUrl
    $ConfigContent | Set-Content $ConfigFile
}

Write-Host "Using Garak virtual environment: $GarakVenv" -ForegroundColor Green
Write-Host "Testing RAG API at: $ApiUrl/query" -ForegroundColor Green
Write-Host "Using config: $ConfigFile" -ForegroundColor Green

# Run garak with REST endpoint (using new command format)
# URI is specified in the JSON config file using -G option
& $GarakExe --target_type rest -G $ConfigFile

