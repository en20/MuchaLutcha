"""Etapa 2 do pipeline: engenharia de atributos, espelhamento e split temporal."""
import numpy as np
import pandas as pd

import config


def _taxa(num, den):
    den = den.replace(0, np.nan)
    return (num / den).fillna(0.0)


def construir_features(df):
    f = pd.DataFrame()
    f["data"]      = df["data"]
    f[config.ALVO] = df["y"]

    # Grupo 1 — diferenciais básicos
    f["d_idade"]        = df["fighter_idade"]       - df["opponent_idade"]
    f["d_altura"]       = df["fighter_altura"]      - df["opponent_altura"]
    f["d_envergadura"]  = df["fighter_envergadura"] - df["opponent_envergadura"]
    f["d_vitorias"]     = df["fighter_vitorias"]    - df["opponent_vitorias"]
    f["d_derrotas"]     = df["fighter_derrotas"]    - df["opponent_derrotas"]
    f["d_lutas_totais"] = df["fighter_lutas_totais"]- df["opponent_lutas_totais"]
    f["d_ko_losses"]    = df["fighter_ko_losses"]   - df["opponent_ko_losses"]

    # Taxas de vitória por método
    for lado in ("fighter", "opponent"):
        v   = df[f"{lado}_vitorias"]
        ko  = df[f"{lado}_ko"]
        sub = df[f"{lado}_sub"]
        dec = (v - ko - sub).clip(lower=0)
        df[f"{lado}_taxa_ko"]  = _taxa(ko,  v)
        df[f"{lado}_taxa_sub"] = _taxa(sub, v)
        df[f"{lado}_taxa_dec"] = _taxa(dec, v)

    f["d_taxa_ko"]  = df["fighter_taxa_ko"]  - df["opponent_taxa_ko"]
    f["d_taxa_sub"] = df["fighter_taxa_sub"] - df["opponent_taxa_sub"]
    f["d_taxa_dec"] = df["fighter_taxa_dec"] - df["opponent_taxa_dec"]

    # Grupo 2 — forma recente
    f["d_L5Y_winrate"] = df["fighter_L5Y_winrate"] - df["opponent_L5Y_winrate"]
    f["d_L2Y_winrate"] = df["fighter_L2Y_winrate"] - df["opponent_L2Y_winrate"]

    # Grupo 3 — estatísticas de luta
    f["d_sig_strikes_landed"]   = df["fighter_sig_strikes_landed"]   - df["opponent_sig_strikes_landed"]
    f["d_sig_strikes_absorbed"] = df["fighter_sig_strikes_absorbed"] - df["opponent_sig_strikes_absorbed"]
    f["d_td_landed"]            = df["fighter_td_landed"]            - df["opponent_td_landed"]
    f["d_td_acc"]               = df["fighter_td_acc"]               - df["opponent_td_acc"]
    f["d_sig_strike_acc"]       = df["fighter_sig_strike_acc"]       - df["opponent_sig_strike_acc"]

    # Grupo 4 — stance
    for e in config.ESTILOS:
        ind_f = (df["fighter_stance"] == e).astype(int)
        ind_o = (df["opponent_stance"] == e).astype(int)
        f[f"d_stance_{e}"] = ind_f - ind_o
    f[config.FEATURE_ESTILOS_DIF] = (
        df["fighter_stance"] != df["opponent_stance"]
    ).astype(int)

    return f


def espelhar(df):
    espelho = df.copy()
    cols_negar = config.FEATURES_DIFF + config.FEATURES_FORMA + config.FEATURES_STATS + config.FEATURES_STANCE
    espelho[cols_negar] = -espelho[cols_negar]
    espelho[config.ALVO] = 1 - espelho[config.ALVO]
    return pd.concat([df, espelho], ignore_index=True)


def gerar():
    config.garantir_diretorios()
    df    = pd.read_csv(config.CSV_PROC, parse_dates=["data"])
    feats = construir_features(df).sort_values("data").reset_index(drop=True)

    n     = len(feats)
    corte = int(n * (1 - config.FRAC_TESTE))
    treino = feats.iloc[:corte].copy()
    teste  = feats.iloc[corte:].copy()

    data_corte = teste["data"].min()
    print(f"Total: {n} lutas | corte em {data_corte.date()}")
    print(f"Treino: {len(treino)} ({treino['data'].dt.year.min()}–{treino['data'].dt.year.max()})")
    print(f"Teste : {len(teste)}  ({teste['data'].dt.year.min()}–{teste['data'].dt.year.max()})")

    treino_esp = espelhar(treino)
    print(f"Treino espelhado: {len(treino_esp)} | classe positiva: {treino_esp[config.ALVO].mean():.3f}")
    print(f"Features totais: {len(config.FEATURES)}")

    treino_esp.to_csv(config.CSV_TREINO, index=False)
    teste.to_csv(config.CSV_TESTE, index=False)
    print(f"\nSalvos: {config.CSV_TREINO}\n        {config.CSV_TESTE}")


if __name__ == "__main__":
    gerar()
