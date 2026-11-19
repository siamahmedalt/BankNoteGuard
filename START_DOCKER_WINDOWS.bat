@echo off
cd /d "%~dp0"
echo Starting BankNoteGuard on http://localhost:5174
where docker >nul 2>nul || (echo Docker not installed & pause & exit /b 1)
docker compose config --quiet || (pause & exit /b 1)
docker compose up --build -d || (pause & exit /b 1)
docker compose ps
start "" "http://localhost:5174"
pause
