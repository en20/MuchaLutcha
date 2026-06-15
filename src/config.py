"""Configurações globais do projeto de previsão de lutas do UFC.

Centraliza caminhos de arquivos, semente aleatória e a lista de atributos
utilizados pelos modelos, para garantir reprodutibilidade entre as etapas
do pipeline (preparação -> features -> treino -> avaliação).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent

DIR_DADOS_BRUTOS = RAIZ / "data" / "raw"
DIR_DADOS_PROC = RAIZ / "data" / "processed"
DIR_RESULTADOS = RAIZ / "results"
DIR_FIGURAS = DIR_RESULTADOS / "figures"
DIR_TABELAS = DIR_RESULTADOS / "tables"
DIR_MODELOS = DIR_RESULTADOS / "models"
DIR_FIG_ARTIGO = RAIZ / "paper" / "figures"

# Arquivos de entrada (copiados do repositório de referência Punch-Prophecy)
CSV_LUTAS = DIR_DADOS_BRUTOS / "ufc_fights.csv"
CSV_LUTADORES = DIR_DADOS_BRUTOS / "fighter_stats.csv"

# Arquivos intermediários
CSV_PROC = DIR_DADOS_PROC / "fights.csv"          # dataset limpo (1 linha por luta)
CSV_TREINO = DIR_DADOS_PROC / "train.csv"          # features + alvo (treino, espelhado)
CSV_TESTE = DIR_DADOS_PROC / "test.csv"            # features + alvo (teste)

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
SEMENTE = 42

# Fração final (mais recente) das lutas reservada para teste (split temporal)
FRAC_TESTE = 0.20

# ---------------------------------------------------------------------------
# Atributos
# ---------------------------------------------------------------------------
# Atributos numéricos diferenciais (lutador - oponente). São os definidos pelo
# grupo: idade, altura, envergadura, cartel (vitórias/derrotas), número de
# lutas anteriores e taxas de vitória por método (nocaute/finalização/decisão).
FEATURES_DIFF = [
    "d_idade",
    "d_altura",
    "d_envergadura",
    "d_vitorias",
    "d_derrotas",
    "d_lutas_totais",
    "d_taxa_ko",
    "d_taxa_sub",
    "d_taxa_dec",
]

# Atributos de estilo de luta (stance). Para cada estilo guardamos a diferença
# de indicadores (lutador - oponente), antissimétrica como as demais features.
ESTILOS = ["orthodox", "southpaw", "switch", "other"]
FEATURES_STANCE = [f"d_stance_{e}" for e in ESTILOS]

# Indicador simétrico de estilos diferentes (não muda ao espelhar a luta).
FEATURE_ESTILOS_DIF = "estilos_diferentes"

# Conjunto completo de features usado pelos modelos.
FEATURES = FEATURES_DIFF + FEATURES_STANCE + [FEATURE_ESTILOS_DIF]

ALVO = "y"  # 1 = o lutador (coluna 'fighter') vence; 0 = perde


def garantir_diretorios():
    """Cria os diretórios de saída caso ainda não existam."""
    for d in (DIR_DADOS_PROC, DIR_FIGURAS, DIR_TABELAS, DIR_MODELOS, DIR_FIG_ARTIGO):
        d.mkdir(parents=True, exist_ok=True)
