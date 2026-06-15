# Executa o pipeline completo e compila o artigo (se tectonic estiver disponivel).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[1/5] Preparacao dos dados"; python src/data_prep.py
Write-Host "[2/5] Engenharia de atributos"; python src/features.py
Write-Host "[3/5] Analise exploratoria"; python src/eda.py
Write-Host "[4/5] Treino dos modelos"; python src/train.py
Write-Host "[5/5] Avaliacao"; python src/evaluate.py

Copy-Item results/tables/metricas_teste.tex paper/metricas_teste.tex -Force
if (Get-Command tectonic -ErrorAction SilentlyContinue) {
    Write-Host "[paper] Compilando com tectonic"
    Push-Location paper; tectonic main.tex; Pop-Location
    Write-Host "PDF gerado em paper/main.pdf"
} else {
    Write-Host "[paper] tectonic nao encontrado; compile paper/main.tex manualmente ou via Overleaf."
}
Write-Host "Concluido."
