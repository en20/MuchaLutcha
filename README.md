# MuchaLutcha — UFC Fight Predictor

Projeto final da disciplina **Aprendizagem de Máquina (2026.1)** — Departamento de Computação, UFC.  
Prof. César Lincoln C. Mattos.

Previsão do vencedor de lutas do UFC como **classificação binária**, comparando cinco modelos de ML sobre **7.827 lutas reais (1993–2024)**. Inclui um web app com scraping ao vivo dos rankings oficiais da UFC.

---

## Resultados

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

---

## Estrutura

```
src/              pipeline de ML (data_prep → features → eda → train → evaluate)
data/raw/         CSVs de entrada (ufc_fights.csv, fighter_stats.csv)
data/processed/   dataset limpo + matrizes de treino/teste
results/          figuras, tabelas e modelos treinados (.joblib)
web/              frontend do web app (index.html)
app.py            servidor Flask (API + web app)
predict.py        preditor interativo via terminal
paper/            artigo em LaTeX
```

---

## Opção 1 — Web App via Docker (recomendado)

> Requer [Docker](https://docs.docker.com/get-docker/) instalado.

### 1. Obter os dados brutos

```bash
git clone --depth=1 --filter=blob:none --sparse \
  https://github.com/leotuckey/Punch-Prophecy-Leo-Tuckey reference
cd reference
git sparse-checkout set src/models/buildingMLModel/data/processed/
cd ..
mkdir -p data/raw
cp reference/src/models/buildingMLModel/data/processed/ufc_fights.csv data/raw/
cp reference/src/models/buildingMLModel/data/processed/fighter_stats.csv data/raw/
```

### 2. Build e run

```bash
docker build -t muchalutcha .
docker run -p 5000:5000 muchalutcha
```

Abra **http://localhost:5000** no navegador.

> O container executa o pipeline completo (treino + avaliação) na primeira inicialização e depois sobe o servidor web. Se quiser pular o treino e usar modelos já treinados, monte o diretório `results/`:
> ```bash
> docker run -p 5000:5000 -v $(pwd)/results:/app/results muchalutcha
> ```

---

## Opção 2 — Rodar localmente

### 1. Dependências

```bash
pip install -r requirements.txt
pip install flask flask-cors requests beautifulsoup4
```

Ou com `uv` (mais rápido):

```bash
uv venv .venv
uv pip install -r requirements.txt
uv pip install flask flask-cors requests beautifulsoup4
```

### 2. Dados

```bash
git clone --depth=1 --filter=blob:none --sparse \
  https://github.com/leotuckey/Punch-Prophecy-Leo-Tuckey reference
cd reference && git sparse-checkout set src/models/buildingMLModel/data/processed/ && cd ..
mkdir -p data/raw
cp reference/src/models/buildingMLModel/data/processed/ufc_fights.csv data/raw/
cp reference/src/models/buildingMLModel/data/processed/fighter_stats.csv data/raw/
```

### 3. Treinar os modelos

```bash
cd src
python data_prep.py   # limpeza → data/processed/fights.csv
python features.py    # engenharia de features + split temporal
python train.py       # GridSearchCV nos 5 modelos (~2 min)
python evaluate.py    # métricas, figuras, tabelas
cd ..
```

Atalho:

```bash
bash run_all.sh       # Linux/macOS
pwsh ./run_all.ps1    # Windows PowerShell
```

### 4. Web App

```bash
python app.py
# → http://localhost:5000
```

### 5. Preditor via terminal (opcional)

```bash
# Demo com 10 lutas históricas reais
python predict.py --demo

# Confronto direto
python predict.py --fight "Jon Jones" "Tom Aspinall"

# Modo interativo
python predict.py
```

---

## Decisões metodológicas

- **Representação diferencial** (lutador − oponente): antissimétrica, permite espelhar cada luta no treino para balancear classes e eliminar viés de canto.
- **Split temporal** (80% antigas / 20% recentes): evita vazamento de dados e simula uso real — treina no passado, testa no futuro.
- **Atributos físicos** preferidos do `fighter_stats.csv` (cobertura ~99,3%); ausentes imputados pela mediana.
- **Seleção de hiperparâmetros** por `GridSearchCV` com 5-fold estratificado, métrica AUC, apenas no conjunto de treino.
- **Semente fixa** `SEMENTE = 42` em `src/config.py` para reprodutibilidade total.

---

## Artigo (PDF)

```bash
cp results/tables/metricas_teste.tex paper/metricas_teste.tex
cd paper
tectonic main.tex   # gera paper/main.pdf
```

Ou suba a pasta `paper/` no [Overleaf](https://overleaf.com).
