"""Etapa 1 do pipeline: limpeza e montagem do dataset de lutas.

Lê os CSVs brutos e produz fights.csv com uma linha por luta.
Aplica corte em ANO_MINIMO (2008) para garantir cobertura adequada
das estatísticas de luta (sig strikes, takedowns, etc.).
"""
import re
import numpy as np
import pandas as pd

import config

CM_POR_POLEGADA = 2.54


def altura_ft_in_para_cm(valor):
    if not isinstance(valor, str):
        return np.nan
    m = re.match(r"\s*(\d+)\s*'\s*(\d+)", valor)
    if not m:
        return np.nan
    return (int(m.group(1)) * 12 + int(m.group(2))) * CM_POR_POLEGADA


def polegadas_para_cm(valor):
    if not isinstance(valor, str):
        return np.nan
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*\"?", valor)
    if not m:
        return np.nan
    return float(m.group(1)) * CM_POR_POLEGADA


def texto_para_numero(serie, ausentes=("unknown", "nan", "")):
    s = serie.astype(str).str.strip().str.lower()
    s = s.where(~s.isin(ausentes), np.nan)
    return pd.to_numeric(s, errors="coerce")


def agrupar_stance(valor):
    if not isinstance(valor, str):
        return "other"
    v = valor.strip().lower()
    return v if v in ("orthodox", "southpaw", "switch") else "other"


def carregar_lutadores():
    s = pd.read_csv(config.CSV_LUTADORES)
    s = s.drop_duplicates(subset="name", keep="first").copy()
    s["altura_cm_fs"]     = s["height"].apply(altura_ft_in_para_cm)
    s["envergadura_cm_fs"]= s["reach"].apply(polegadas_para_cm)
    s["stance_fs"]        = s["stance"].apply(agrupar_stance)
    return s[["name", "altura_cm_fs", "envergadura_cm_fs", "stance_fs"]]


def safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    """num/den with 0 when den==0 (no divide-by-zero)."""
    d = den.replace(0, np.nan)
    return (num / d).fillna(0.0)


def preparar():
    config.garantir_diretorios()
    df = pd.read_csv(config.CSV_LUTAS, low_memory=False)
    print(f"Linhas brutas: {len(df)}")

    df["data"] = pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce")
    df = df.dropna(subset=["data"]).sort_values("data").reset_index(drop=True)

    # Corte temporal: só lutas a partir de ANO_MINIMO
    df = df[df["data"].dt.year >= config.ANO_MINIMO].copy()
    print(f"Após corte {config.ANO_MINIMO}+: {len(df)} linhas")

    # Alvo binário
    df = df[df["result"].isin(["W", "L"])].copy()
    df["y"] = (df["result"] == "W").astype(int)
    print(f"Após filtrar W/L: {len(df)}  | taxa vitória: {df['y'].mean():.3f}")

    lut = carregar_lutadores()

    saida = pd.DataFrame({
        "data":     df["data"].values,
        "fighter":  df["fighter"].values,
        "opponent": df["opponent"].values,
        "y":        df["y"].values,
        "method":   df["method"].values,
        "division": df["division"].values,
    })

    for lado, prefixo in [("fighter", "fighter"), ("opponent", "opponent")]:
        # --- atributos de cartel ---
        vitorias = pd.to_numeric(df[f"{prefixo}_wins"],   errors="coerce")
        derrotas = pd.to_numeric(df[f"{prefixo}_losses"], errors="coerce")
        ko_w     = pd.to_numeric(df[f"{prefixo}_ko_wins"],    errors="coerce").fillna(0)
        sub_w    = pd.to_numeric(df[f"{prefixo}_sub_wins"],   errors="coerce").fillna(0)
        ko_l     = pd.to_numeric(df[f"{prefixo}_ko_losses"],  errors="coerce").fillna(0)

        # --- forma recente ---
        L5Y_w = pd.to_numeric(df[f"{prefixo}_L5Y_wins"],   errors="coerce").fillna(0)
        L5Y_l = pd.to_numeric(df[f"{prefixo}_L5Y_losses"], errors="coerce").fillna(0)
        L2Y_w = pd.to_numeric(df[f"{prefixo}_L2Y_wins"],   errors="coerce").fillna(0)
        L2Y_l = pd.to_numeric(df[f"{prefixo}_L2Y_losses"], errors="coerce").fillna(0)

        # --- estatísticas de luta (médias pré-luta) ---
        sig_l  = pd.to_numeric(df[f"{prefixo}_inf_sig_strikes_landed_avg"],   errors="coerce").fillna(0)
        sig_a  = pd.to_numeric(df[f"{prefixo}_inf_sig_strikes_attempts_avg"], errors="coerce").fillna(0)
        sig_ab = pd.to_numeric(df[f"{prefixo}_abs_sig_strikes_landed_avg"],   errors="coerce").fillna(0)
        td_l   = pd.to_numeric(df[f"{prefixo}_inf_takedowns_landed_avg"],     errors="coerce").fillna(0)
        td_a   = pd.to_numeric(df[f"{prefixo}_inf_takedowns_attempts_avg"],   errors="coerce").fillna(0)

        # --- físicos (fighter_stats.csv preferido) ---
        idade_raw = texto_para_numero(df[f"{prefixo}_age"],
                                      ausentes=("unknown","nan","","0","0.0"))
        altura_uf = pd.to_numeric(df[f"{prefixo}_height"], errors="coerce")
        env_uf    = pd.to_numeric(df[f"{prefixo}_reach"],  errors="coerce")

        info     = df[[lado]].merge(lut, left_on=lado, right_on="name", how="left")
        altura   = pd.Series(info["altura_cm_fs"].values).fillna(altura_uf.reset_index(drop=True))
        envergad = pd.Series(info["envergadura_cm_fs"].values).fillna(env_uf.reset_index(drop=True))
        stance   = pd.Series(info["stance_fs"].values).fillna("other")

        # Imputar físicos pela mediana
        for col_name, serie in [("idade", idade_raw), ("altura", altura), ("envergadura", envergad)]:
            mediana = serie.median()
            n_miss  = serie.isna().sum()
            serie   = serie.fillna(mediana)
            if n_miss:
                print(f"  imputados {n_miss:4d} em {lado}_{col_name} (mediana={mediana:.1f})")
            if col_name == "idade":     saida[f"{lado}_idade"]       = serie.values
            elif col_name == "altura":  saida[f"{lado}_altura"]      = serie.values
            else:                       saida[f"{lado}_envergadura"]  = serie.values

        saida[f"{lado}_vitorias"]      = vitorias.values
        saida[f"{lado}_derrotas"]      = derrotas.values
        saida[f"{lado}_lutas_totais"]  = (vitorias + derrotas).values
        saida[f"{lado}_ko"]            = ko_w.values
        saida[f"{lado}_sub"]           = sub_w.values
        saida[f"{lado}_ko_losses"]     = ko_l.values
        saida[f"{lado}_L5Y_winrate"]   = safe_rate(L5Y_w, L5Y_w + L5Y_l).values
        saida[f"{lado}_L2Y_winrate"]   = safe_rate(L2Y_w, L2Y_w + L2Y_l).values
        saida[f"{lado}_sig_strikes_landed"]   = sig_l.values
        saida[f"{lado}_sig_strikes_absorbed"] = sig_ab.values
        saida[f"{lado}_td_landed"]     = td_l.values
        saida[f"{lado}_td_acc"]        = safe_rate(td_l, td_a).values
        saida[f"{lado}_sig_strike_acc"]= safe_rate(sig_l, sig_a).values
        saida[f"{lado}_stance"]        = stance.values

    saida = saida.dropna(subset=[
        "fighter_vitorias", "opponent_vitorias",
        "fighter_derrotas", "opponent_derrotas",
    ]).reset_index(drop=True)

    saida.to_csv(config.CSV_PROC, index=False)
    print(f"\nSalvo: {config.CSV_PROC}  | {len(saida)} lutas, "
          f"{pd.to_datetime(saida['data']).dt.year.min()}–{pd.to_datetime(saida['data']).dt.year.max()}")
    return saida


if __name__ == "__main__":
    preparar()
