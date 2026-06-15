"""Etapa 1 do pipeline: limpeza e montagem do dataset de lutas.

Lê os CSVs brutos do repositório de referência (Punch-Prophecy) e produz um
arquivo limpo com uma linha por luta (`data/processed/fights.csv`), contendo,
para o lutador e o oponente, os atributos definidos pelo grupo:

    idade, altura, envergadura, vitórias, derrotas, número de lutas,
    estilo de luta (stance) e contagem de vitórias por nocaute/finalização.

Decisões de limpeza importantes:
  * As colunas de idade/altura/envergadura do `ufc_fights.csv` vêm como texto e
    usam ``'unknown'`` (e ``'0'`` para idade) como marcador de ausência.
  * Altura, envergadura e estilo são obtidos preferencialmente do
    `fighter_stats.csv` (texto limpo, ~99% de cobertura por nome), recorrendo às
    colunas de `ufc_fights.csv` apenas como reserva.
  * Empates (``result == 'D'``) e resultados anulados são descartados, pois o
    problema é de classificação binária (vitória/derrota).
"""
import re
import numpy as np
import pandas as pd

import config

CM_POR_POLEGADA = 2.54


# ---------------------------------------------------------------------------
# Conversores de unidade
# ---------------------------------------------------------------------------
def altura_ft_in_para_cm(valor):
    """Converte alturas no formato ``6' 4"`` para centímetros."""
    if not isinstance(valor, str):
        return np.nan
    m = re.match(r"\s*(\d+)\s*'\s*(\d+)", valor)
    if not m:
        return np.nan
    pes, pol = int(m.group(1)), int(m.group(2))
    return (pes * 12 + pol) * CM_POR_POLEGADA


def polegadas_para_cm(valor):
    """Converte envergaduras no formato ``79"`` para centímetros."""
    if not isinstance(valor, str):
        return np.nan
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*\"?", valor)
    if not m:
        return np.nan
    return float(m.group(1)) * CM_POR_POLEGADA


def texto_para_numero(serie, ausentes=("unknown", "nan", "")):
    """Converte uma série de texto em float, mapeando marcadores para NaN."""
    s = serie.astype(str).str.strip().str.lower()
    s = s.where(~s.isin(ausentes), np.nan)
    return pd.to_numeric(s, errors="coerce")


def agrupar_stance(valor):
    """Normaliza o estilo de luta em {orthodox, southpaw, switch, other}."""
    if not isinstance(valor, str):
        return "other"
    v = valor.strip().lower()
    if v in ("orthodox", "southpaw", "switch"):
        return v
    return "other"  # Open Stance, Sideways, ausente, etc.


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------
def carregar_lutadores():
    """Tabela auxiliar com altura(cm), envergadura(cm) e estilo por lutador."""
    s = pd.read_csv(config.CSV_LUTADORES)
    s = s.drop_duplicates(subset="name", keep="first").copy()
    s["altura_cm_fs"] = s["height"].apply(altura_ft_in_para_cm)
    s["envergadura_cm_fs"] = s["reach"].apply(polegadas_para_cm)
    s["stance_fs"] = s["stance"].apply(agrupar_stance)
    return s[["name", "altura_cm_fs", "envergadura_cm_fs", "stance_fs"]]


def preparar():
    config.garantir_diretorios()
    df = pd.read_csv(config.CSV_LUTAS, low_memory=False)
    print(f"Linhas brutas: {len(df)}")

    # Data e ordenação cronológica
    df["data"] = pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce")
    df = df.dropna(subset=["data"]).sort_values("data").reset_index(drop=True)

    # Alvo binário: mantém apenas vitória/derrota do ponto de vista do 'fighter'
    df = df[df["result"].isin(["W", "L"])].copy()
    df["y"] = (df["result"] == "W").astype(int)
    print(f"Linhas após filtrar W/L: {len(df)}  | taxa de vitória: {df['y'].mean():.3f}")

    lut = carregar_lutadores()

    saida = pd.DataFrame({
        "data": df["data"].values,
        "fighter": df["fighter"].values,
        "opponent": df["opponent"].values,
        "y": df["y"].values,
        "method": df["method"].values,
    })

    for lado, prefixo in [("fighter", "fighter"), ("opponent", "opponent")]:
        # Atributos dinâmicos (vêm do ufc_fights, pré-luta)
        idade = texto_para_numero(df[f"{prefixo}_age"], ausentes=("unknown", "nan", "", "0", "0.0"))
        altura_uf = texto_para_numero(df[f"{prefixo}_height"])
        env_uf = texto_para_numero(df[f"{prefixo}_reach"])
        vitorias = pd.to_numeric(df[f"{prefixo}_wins"], errors="coerce")
        derrotas = pd.to_numeric(df[f"{prefixo}_losses"], errors="coerce")
        ko = pd.to_numeric(df[f"{prefixo}_ko_wins"], errors="coerce")
        sub = pd.to_numeric(df[f"{prefixo}_sub_wins"], errors="coerce")

        # Atributos estáticos (vêm do fighter_stats, mais limpos)
        info = df[[lado]].merge(lut, left_on=lado, right_on="name", how="left")
        altura = info["altura_cm_fs"].values
        envergadura = info["envergadura_cm_fs"].values
        stance = info["stance_fs"].fillna("other").values

        # Reserva: usa o ufc_fights quando o fighter_stats não tem o valor
        altura = pd.Series(altura).fillna(altura_uf.reset_index(drop=True))
        envergadura = pd.Series(envergadura).fillna(env_uf.reset_index(drop=True))

        saida[f"{lado}_idade"] = idade.values
        saida[f"{lado}_altura"] = altura.values
        saida[f"{lado}_envergadura"] = envergadura.values
        saida[f"{lado}_vitorias"] = vitorias.values
        saida[f"{lado}_derrotas"] = derrotas.values
        saida[f"{lado}_lutas_totais"] = (vitorias + derrotas).values
        saida[f"{lado}_ko"] = ko.values
        saida[f"{lado}_sub"] = sub.values
        saida[f"{lado}_stance"] = stance

    # Imputação dos numéricos contínuos (idade/altura/envergadura) pela mediana
    for col in saida.columns:
        if any(k in col for k in ("idade", "altura", "envergadura")):
            mediana = saida[col].median()
            n_faltando = saida[col].isna().sum()
            saida[col] = saida[col].fillna(mediana)
            if n_faltando:
                print(f"  imputados {n_faltando:4d} valores em {col} (mediana={mediana:.1f})")

    saida = saida.dropna(subset=[
        "fighter_vitorias", "opponent_vitorias",
        "fighter_derrotas", "opponent_derrotas",
    ]).reset_index(drop=True)

    config.CSV_PROC.parent.mkdir(parents=True, exist_ok=True)
    saida.to_csv(config.CSV_PROC, index=False)
    print(f"\nDataset salvo em {config.CSV_PROC}  | {len(saida)} lutas, "
          f"{saida['data'].dt.year.min()}-{saida['data'].dt.year.max()}")
    return saida


if __name__ == "__main__":
    preparar()
