"""Configurações globais do projeto de previsão de lutas do UFC."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent

DIR_DADOS_BRUTOS = RAIZ / "data" / "raw"
DIR_DADOS_PROC   = RAIZ / "data" / "processed"
DIR_RESULTADOS   = RAIZ / "results"
DIR_FIGURAS      = DIR_RESULTADOS / "figures"
DIR_TABELAS      = DIR_RESULTADOS / "tables"
DIR_MODELOS      = DIR_RESULTADOS / "models"
DIR_FIG_ARTIGO   = RAIZ / "paper" / "figures"

CSV_LUTAS     = DIR_DADOS_BRUTOS / "ufc_fights.csv"
CSV_LUTADORES = DIR_DADOS_BRUTOS / "fighter_stats.csv"

CSV_PROC   = DIR_DADOS_PROC / "fights.csv"
CSV_TREINO = DIR_DADOS_PROC / "train.csv"
CSV_TESTE  = DIR_DADOS_PROC / "test.csv"

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
SEMENTE    = 42
FRAC_TESTE = 0.20

# Corte temporal: ignorar lutas antes de 2008 (stats de luta pouco cobertas).
ANO_MINIMO = 2008

# ---------------------------------------------------------------------------
# Atributos
# ---------------------------------------------------------------------------

# Fator de amplificação da taxa de decisão em lutas de título (5 rounds)
TITLE_FIGHT_DEC_BOOST = 3.0

# Grupo 1 — diferenciais básicos (lutador - oponente)
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
    "d_ko_losses",          # chin: derrotas por nocaute é proxy de queixo ruim
    "d_dec_rate_overall",   # % de lutas (W+L) que foram a decisão — proxy de resistência
]

# Grupo 2 — forma recente (últimas 5 e 2 lutas)
FEATURES_FORMA = [
    "d_L5Y_winrate",    # taxa de vitória nos últimos 5 anos
    "d_L2Y_winrate",    # taxa de vitória nos últimos 2 anos (mais sensível)
]

# Grupo 3 — estatísticas de luta (médias históricas pré-luta)
FEATURES_STATS = [
    "d_sig_strikes_landed",   # golpes significativos dados/luta
    "d_sig_strikes_absorbed", # golpes significativos levados/luta
    "d_td_landed",            # takedowns concluídos/luta
    "d_td_acc",               # precisão de takedown
    "d_sig_strike_acc",       # precisão de golpes significativos
]

# Grupo 4 — stance
ESTILOS = ["orthodox", "southpaw", "switch", "other"]
FEATURES_STANCE = [f"d_stance_{e}" for e in ESTILOS]
FEATURE_ESTILOS_DIF = "estilos_diferentes"

# Conjunto completo
FEATURES = (
    FEATURES_DIFF
    + FEATURES_FORMA
    + FEATURES_STATS
    + FEATURES_STANCE
    + [FEATURE_ESTILOS_DIF]
)

ALVO = "y"  # 1 = lutador vence; 0 = perde


def garantir_diretorios():
    for d in (DIR_DADOS_PROC, DIR_FIGURAS, DIR_TABELAS, DIR_MODELOS, DIR_FIG_ARTIGO):
        d.mkdir(parents=True, exist_ok=True)
