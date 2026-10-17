# ==============================================================================
# ⚡ SPARK-DECIDE — Download & Setup Veloce 1-Click
# ==============================================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Clear-Host

Write-Host ""
Write-Host "  ==========================================================================" -ForegroundColor DarkGray
Write-Host "    ⚡ SPARK-DECIDE — DOWNLOADER & INSTALLATORE VELOCE" -ForegroundColor Green
Write-Host "    Motore Decisionale Strutturato a Singolo Forward Pass" -ForegroundColor White
Write-Host "  ==========================================================================" -ForegroundColor DarkGray
Write-Host ""

# Selettore cartella con interfaccia grafica moderna
Add-Type -AssemblyName System.Windows.Forms
$FolderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
$FolderBrowser.Description = "Seleziona la cartella dove desideri scaricare ed eseguire Spark-Decide:"
$FolderBrowser.SelectedPath = (Get-Location).Path
$FolderBrowser.ShowNewFolderButton = $true

Write-Host "  [?] Apertura selettore cartella in corso..." -ForegroundColor Yellow
$result = $FolderBrowser.ShowDialog()

if ($result -ne [System.Windows.Forms.DialogResult]::OK -or [string]::IsNullOrWhiteSpace($FolderBrowser.SelectedPath)) {
    Write-Host "  [-] Nessuna cartella selezionata. Utilizzo cartella corrente: $((Get-Location).Path)" -ForegroundColor Gray
    $targetDir = (Get-Location).Path
} else {
    $targetDir = $FolderBrowser.SelectedPath
}

if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
}

Set-Location $targetDir

Write-Host "  [+] Cartella selezionata: $targetDir" -ForegroundColor Cyan
Write-Host ""

$exePath = Join-Path $targetDir "SparkDecide_Setup.exe"
$exeUrl = "https://raw.githubusercontent.com/maxpy9/spark-decide/main/SparkDecide_Setup.exe"

Write-Host "  [↓] Download rapido di SparkDecide_Setup.exe in corso..." -ForegroundColor Green
$webClient = New-Object System.Net.WebClient
$webClient.Headers.Add("User-Agent", "Spark-Decide-Downloader/2.0")

try {
    $webClient.DownloadFile($exeUrl, $exePath)
    Write-Host "  [✓] Download completato con successo!" -ForegroundColor Green
    Write-Host "  [▶] Avvio di Spark-Decide Setup..." -ForegroundColor Cyan
    Write-Host ""
    Start-Process -FilePath $exePath
} catch {
    Write-Host "  [!] Errore nel download: $_" -ForegroundColor Red
    Write-Host "  Puoi scaricarlo manualmente da: $exeUrl" -ForegroundColor Gray
    pause
}
