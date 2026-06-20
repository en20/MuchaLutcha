# MuchaLutcha — UFC Fight Predictor

Projeto final da disciplina **Aprendizagem de Máquina (2026.1)** — Departamento de Computação, UFC.  
Prof. César Lincoln C. Mattos.

Previsão do vencedor de lutas do UFC como **classificação binária**, comparando cinco modelos de ML sobre **9.479 lutas reais (1993–2026)**. Inclui um web app com scraping ao vivo dos rankings oficiais da UFC e confrontos livres entre qualquer categoria de peso.

---

## Resultados

| Modelo | AUC-ROC (CV) |
|---|---|
| **Regressão Logística** | **0,637** |
| XGBoost | 0,635 |
| Random Forest | 0,629 |
| Rede Neural (MLP) | 0,624 |
| SVM (RBF) | 0,622 |
| Baseline (classe majoritária) | ~0,500 |

Atributos mais determinantes: **diferença de vitórias/derrotas**, **forma recente (L5Y/L2Y)** e **taxas de finalização por método**.

---

## Web App

Interface completa para simular confrontos entre lutadores reais ou criados pelo usuário:

- **Classic mode** — seleciona lutadores de qualquer divisão, com aviso quando cruza categorias
- **(Open)heimmer mode** — qualquer lutador vs qualquer lutador, sem restrições
- **Picker com filtro de divisão** — ao selecionar um fighter, escolha a categoria desejada no dropdown; suporte a "Todas as divisões" para busca livre
- **Custom Fighter** — crie um lutador fictício com stats completas e teste contra lutadores reais (acessível pelo menu no canto superior direito)
- **Fotos dos atletas** — carregadas assincronamente da UFC.com
- **Title Fight toggle** — alterna entre 3 e 5 rounds; fighters com maior taxa de lutas por decisão recebem boost no algoritmo em lutas de título
- **Previsão em ensemble** — média dos 5 modelos com detalhamento por modelo e comparativo de atributos

---

## Features do modelo (23 atributos diferenciais)

| Grupo | Atributos |
|---|---|
| Básico | idade, altura, envergadura, vitórias, derrotas, lutas totais |
| Método | taxa KO, taxa SUB, taxa DEC, KO losses, dec_rate_overall |
| Forma recente | winrate nos últimos 5 anos, winrate nos últimos 2 anos |
| Strike stats | sig. strikes dados/luta, absorvidos/luta, takedowns, td_acc, strike_acc |
| Stance | orthodox, southpaw, switch, other (indicadores diferenciais) |

`dec_rate_overall` = % de todas as lutas (W+L) que foram a decisão — proxy de resistência para 5 rounds. Em title fights, esse atributo recebe um boost de 3× no momento da inferência.

---

## Estrutura

```
src/              pipeline de ML (data_prep → features → eda → train → evaluate)
data/raw/         CSVs de entrada (ufc_fights.csv, fighter_stats.csv, fighter_cache.json)
data/processed/   dataset limpo + matrizes de treino/teste
results/          figuras, tabelas e modelos treinados (.joblib)
scripts/          utilitários (update_dataset.py — scraping de novos eventos)
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
pip install flask flask-cors requests beautifulsoup4 playwright
playwright install chromium
```

Ou com `uv`:

```bash
uv venv .venv
uv pip install -r requirements.txt
uv pip install flask flask-cors requests beautifulsoup4 playwright
playwright install chromium
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
python train.py       # GridSearchCV nos 5 modelos (~5 min)
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

### 5. Atualizar o dataset com eventos recentes (opcional)

O script `scripts/update_dataset.py` raspa automaticamente todos os eventos do ufcstats.com após o corte atual do dataset e adiciona as novas lutas ao CSV. Requer Playwright (Chromium) para contornar o Cloudflare.

```bash
python scripts/update_dataset.py
# Após o scraping, re-rodar o pipeline:
cd src && python data_prep.py && python features.py && python train.py && cd ..
```

Os stats de fighter do scraping são extraídos da página atual de cada atleta no ufcstats.com (vitórias/derrotas, físico, médias de carreira). Os resultados são cacheados em `data/raw/fighter_cache.json` para reutilização em futuras execuções.

### 6. Preditor via terminal (opcional)

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
- **Split temporal** (80% antigas / 20% recentes): evita vazamento de dados e simula uso real — treina no passado, testa no futuro. O conjunto de teste cobre 2024–2026.
- **Atributos físicos** preferidos do `fighter_stats.csv` (cobertura ~99,3%); ausentes imputados pela mediana.
- **Seleção de hiperparâmetros** por `GridSearchCV` com 5-fold estratificado, métrica AUC, apenas no conjunto de treino.
- **Boost de título fight**: em lutas de 5 rounds, o atributo `d_dec_rate_overall` é multiplicado por 3× na inferência, favorecendo fighters com histórico de ir a decisão — proxy de resistência aeróbica.
- **Scraping com Playwright**: ufcstats.com usa Cloudflare com desafio JavaScript; requisições simples (curl/requests) retornam página de bloqueio. O Playwright executa um browser Chromium headless que resolve o desafio automaticamente.
- **Semente fixa** `SEMENTE = 42` em `src/config.py` para reprodutibilidade total.

---

## Artigo (PDF)

```bash
cp results/tables/metricas_teste.tex paper/metricas_teste.tex
cd paper
tectonic main.tex   # gera paper/main.pdf
```

Ou suba a pasta `paper/` no [Overleaf](https://overleaf.com).
