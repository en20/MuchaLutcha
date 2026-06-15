# Previsão de Resultados de Lutas do UFC com Aprendizagem de Máquina

Projeto final da disciplina **Aprendizagem de Máquina (2026.1)** — Departamento
de Computação, UFC. Prof. César Lincoln C. Mattos.

Formulamos a previsão do vencedor de uma luta do UFC como **classificação
binária** e comparamos cinco modelos de ML (Regressão Logística, Random Forest,
XGBoost, SVM e Rede Neural/MLP) sobre **7.827 lutas reais (1993–2024)**.

## Resultados principais

| Modelo | Acurácia | AUC-ROC |
|---|---|---|
| **Random Forest** | **0,602** | **0,637** |
| Rede Neural (MLP) | 0,596 | 0,631 |
| Regressão Logística | 0,597 | 0,630 |
| XGBoost | 0,584 | 0,629 |
| SVM (RBF) | 0,596 | 0,623 |
| Baseline (melhor cartel) | 0,492 | 0,486 |
| Baseline (classe majoritária) | 0,497 | — |

Atributos mais determinantes: **diferença de idade** e **diferença de derrotas**.

## Estrutura

```
data/raw/         CSVs reais (do repositório Punch-Prophecy)
data/processed/   dataset limpo + matrizes de treino/teste
src/              pipeline em Python
results/          figuras, tabelas e modelos treinados
paper/            artigo em LaTeX (main.tex -> main.pdf)
```

## Como reproduzir

### 1. Dependências
```bash
pip install -r requirements.txt
```

### 2. Dados
Os CSVs já estão em `data/raw/`. Caso precise reobtê-los, clone o repositório de
referência e copie os arquivos:
```bash
git clone https://github.com/leotuckey/Punch-Prophecy-Leo-Tuckey reference
cp reference/src/models/buildingMLModel/data/processed/ufc_fights.csv data/raw/
cp reference/src/models/buildingMLModel/data/processed/fighter_stats.csv data/raw/
```

### 3. Pipeline
Execute as etapas em ordem (ou use o script `run_all`):
```bash
cd src
python data_prep.py   # limpeza -> data/processed/fights.csv
python features.py     # features + espelhamento + split temporal
python eda.py          # figuras exploratórias
python train.py        # treina e ajusta os 5 modelos (GridSearchCV)
python evaluate.py     # métricas, ROC, importância, calibração
```

Atalho (na raiz do projeto):
```bash
bash run_all.sh          # Linux/Git Bash
# ou
pwsh ./run_all.ps1       # Windows PowerShell
```

### 4. Artigo (PDF)
Requer uma distribuição LaTeX. Recomendamos o
[tectonic](https://tectonic-typesetting.github.io/) (binário único):
```bash
cp results/tables/metricas_teste.tex paper/metricas_teste.tex
cd paper
tectonic main.tex        # gera paper/main.pdf
```
Alternativamente, suba a pasta `paper/` no [Overleaf](https://overleaf.com).

## Decisões metodológicas

- **Fonte limpa de atributos físicos**: altura, envergadura e estilo vêm do
  `fighter_stats.csv` (texto limpo, 99,3% de cobertura), usando `ufc_fights.csv`
  só como reserva; ausentes contínuos imputados pela mediana.
- **Representação diferencial** (lutador − oponente), antissimétrica.
- **Espelhamento** do treino: remove viés de canto e balanceia as classes.
- **Divisão temporal** (80% treino mais antigo / 20% teste mais recente): evita
  vazamento e simula o uso real.
- **Seleção por validação cruzada** (5 folds, AUC) apenas no treino.

## Reprodutibilidade
Semente aleatória fixa (`SEMENTE = 42` em `src/config.py`).
