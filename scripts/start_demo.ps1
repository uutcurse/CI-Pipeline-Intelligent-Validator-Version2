Write-Host "=== CI Workflow Risk Validator Demo ===" -ForegroundColor Cyan
Write-Host "Verifying Python environment..."
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Python .venv not found. Please setup first." -ForegroundColor Red
    exit 1
}

Write-Host "Starting FastAPI Backend (Port 8080)..." -ForegroundColor Yellow
$backend = Start-Process ".\.venv\Scripts\python.exe" -ArgumentList "scripts/run_api.py" -PassThru -NoNewWindow

Start-Sleep -Seconds 3
Write-Host "Backend started."

Write-Host "Starting React Frontend (Vite)..." -ForegroundColor Yellow
$frontend = Start-Process "npm.cmd" -ArgumentList "run dev" -WorkingDirectory "frontend" -PassThru -NoNewWindow

Write-Host "
All systems launching!" -ForegroundColor Green
Write-Host "Backend API available at: http://127.0.0.1:8080"
Write-Host "Frontend UI available at: http://localhost:5173"
Write-Host "Press any key to gracefully shut down the servers..."

$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
Write-Host "Demo shutdown complete."
