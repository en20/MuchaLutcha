#!/usr/bin/env bash
# Executa o pipeline completo e compila o artigo (se tectonic estiver disponível).
set -e
cd "$(dirname "$0")"

echo "[1/5] Preparacao dos dados"; python src/data_prep.py
echo "[2/5] Engenharia de atributos"; python src/features.py
echo "[3/5] Analise exploratoria"; python src/eda.py
echo "[4/5] Treino dos modelos";   python src/train.py
echo "[5/5] Avaliacao";            python src/evaluate.py

cp results/tables/metricas_teste.tex paper/metricas_teste.tex
if command -v tectonic >/dev/null 2>&1; then
  echo "[paper] Compilando com tectonic"; (cd paper && tectonic main.tex)
  echo "PDF gerado em paper/main.pdf"
else
  echo "[paper] tectonic nao encontrado; compile paper/main.tex manualmente ou via Overleaf."
fi
echo "Concluido."
