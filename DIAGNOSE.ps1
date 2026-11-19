Set-Location $PSScriptRoot
Write-Host "=== BankNoteGuard ==="
docker compose ps -a
Write-Host "=== AI health ==="
try { Invoke-RestMethod "http://localhost:8001/health" | ConvertTo-Json -Depth 6 } catch { Write-Host $_.Exception.Message }
Write-Host "=== Java backend health ==="
try { Invoke-RestMethod "http://localhost:8081/api/system" | ConvertTo-Json -Depth 6 } catch { Write-Host $_.Exception.Message }
Write-Host "=== Browser frontend ==="
try { (Invoke-WebRequest -UseBasicParsing "http://localhost:5174").StatusCode } catch { Write-Host $_.Exception.Message }
Write-Host "=== Logs: AI ==="
docker compose logs --tail=80 ai
Write-Host "=== Logs: Backend ==="
docker compose logs --tail=80 backend
Write-Host "=== Logs: Web ==="
docker compose logs --tail=50 web
