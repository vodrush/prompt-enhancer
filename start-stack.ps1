# Auto-start du stack complet au logon :
#   1. OmniRoute natif Windows (daemon, port 20128) — si pas deja en cours
#   2. Watcher (Docker, port 20500) — attend le moteur Docker, puis compose up
# Idempotent : sans effet si tout tourne deja.
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\rapha\hermes-watcher"

# --- 1. OmniRoute natif -----------------------------------------------------
$omnirouteCmd = "C:\Users\rapha\AppData\Roaming\npm\omniroute.cmd"
$listening = Get-NetTCPConnection -LocalPort 20128 -State Listen -ErrorAction SilentlyContinue
if (-not $listening) {
    Write-Output "[stack] OmniRoute pas en cours -> lancement daemon"
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$omnirouteCmd`" serve --daemon" -WindowStyle Hidden
    for ($i = 0; $i -lt 24; $i++) {
        Start-Sleep -Seconds 2
        if (Get-NetTCPConnection -LocalPort 20128 -State Listen -ErrorAction SilentlyContinue) { break }
    }
} else {
    Write-Output "[stack] OmniRoute deja en cours"
}

# --- 2. Watcher Docker ------------------------------------------------------
$ready = $false
for ($i = 0; $i -lt 24; $i++) {
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 5
}
if (-not $ready) { Write-Error "Moteur Docker non pret apres 120s"; exit 1 }

docker compose up -d --build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
docker compose ps

Write-Output "[stack] OmniRoute : http://localhost:20128 | Watcher : http://localhost:20500"
