# PowerShell script to run Locust load tests with automatic result saving
# Usage: .\run_locust_load_test.ps1 -Users 50 -SpawnRate 10 -RunTime "5m"

param(
    [int]$Users = 12,
    [int]$SpawnRate = 1,
    [string]$RunTime = "2m",
    [string]$Host = "http://localhost:8000",
    [switch]$Headless = $true
)

# Get timestamp for unique file names
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Set report directory (relative to script location)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$reportsDir = Join-Path $projectRoot "reports"
$performanceReportsDir = Join-Path $reportsDir "performance"

# Ensure reports directories exist
if (-not (Test-Path $reportsDir)) {
    New-Item -ItemType Directory -Path $reportsDir | Out-Null
}
if (-not (Test-Path $performanceReportsDir)) {
    New-Item -ItemType Directory -Path $performanceReportsDir | Out-Null
}

# Build CSV and HTML file paths (save to performance subdirectory)
$csvPrefix = Join-Path $performanceReportsDir "locust_load_$timestamp"
$htmlFile = Join-Path $performanceReportsDir "locust_load_$timestamp.html"
$logFile = Join-Path $performanceReportsDir "locust_load_$timestamp.log"

# Build Locust command
$locustFile = Join-Path $projectRoot "tests\performance\locustLoad.py"
$cmd = "locust -f `"$locustFile`" --host=$Host -u $Users -r $SpawnRate --run-time $RunTime"

if ($Headless) {
    $cmd += " --headless"
}

$cmd += " --csv=`"$csvPrefix`""
$cmd += " --html=`"$htmlFile`""
$cmd += " --logfile=`"$logFile`""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Locust Load Test Configuration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Users: $Users" -ForegroundColor Yellow
Write-Host "Spawn Rate: $SpawnRate users/second" -ForegroundColor Yellow
Write-Host "Run Time: $RunTime" -ForegroundColor Yellow
Write-Host "Host: $Host" -ForegroundColor Yellow
Write-Host "Headless: $Headless" -ForegroundColor Yellow
Write-Host ""
Write-Host "Results will be saved to:" -ForegroundColor Green
Write-Host "  CSV: $csvPrefix*.csv" -ForegroundColor Green
Write-Host "  HTML: $htmlFile" -ForegroundColor Green
Write-Host "  Log: $logFile" -ForegroundColor Green
Write-Host ""
Write-Host "Running Locust..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Run Locust
Invoke-Expression $cmd

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Test completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Results saved to:" -ForegroundColor Green
Write-Host "  CSV files: $csvPrefix*.csv" -ForegroundColor Green
Write-Host "  HTML report: $htmlFile" -ForegroundColor Green
Write-Host "  Log file: $logFile" -ForegroundColor Green

