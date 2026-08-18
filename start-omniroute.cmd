@echo off
REM ============================================================
REM  Auto-start OmniRoute (natif Windows) au demarrage.
REM  Lance en arriere-plan SANS fenetre (windowstyle hidden).
REM  Idempotent : sort si le port 20128 est deja occupe.
REM ============================================================

netstat -an | findstr "LISTENING" | findstr ":20128" >nul 2>&1
if not errorlevel 1 (
    exit /b 0
)

powershell -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath 'C:\Users\rapha\AppData\Roaming\npm\omniroute.cmd' -ArgumentList 'serve','--daemon' -WindowStyle Hidden"

exit /b 0
