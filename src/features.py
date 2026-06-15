"""Etapa 2 do pipeline: engenharia de atributos, espelhamento e split temporal.

A partir do dataset limpo (`fights.csv`) produz a matriz de atributos usada
pelos modelos. Cada luta é representada por **diferenças** entre o lutador e o
oponente (lutador - oponente), o que torna a representação antissimétrica:
inverter os dois cantos equivale a negar todas as features e o rótulo.

Saídas:
  * `data/processed/train.csv`: lutas mais antigas (~80%), **espelhadas**
    (cada luta entra duas vezes, com os cantos trocados) para impor simetria e
    balanceamento exato das classes.
  * `data/processed/test.csv`: lutas mais recentes (~20%), sem espelhamento,
    refletindo o uso real (prever eventos futuros).
"""
import numpy as np
import pandas as pd

import config


def _taxa(num, den):
    """Razão segura num/den; retorna 0 quando o lutador não tem vitórias."""
    den = den.replace(0, np.nan)
    return (num / den).fillna(0.0)


def construir_features(df):
    """Constrói o DataFrame de features diferenciais a partir das lutas limpas."""
    f = pd.DataFrame()
    f["data"] = df["data"]
    f[config.ALVO] = df["y"]

    # Diferenças numéricas diretas (lutador - oponente)
    f["d_idade"] = df["fighter_idade"] - df["opponent_idade"]
    f["d_altura"] = df["fighter_altura"] - df["opponent_altura"]
    f["d_envergadura"] = df["fighter_envergadura"] - df["opponent_envergadura"]
    f["d_vitorias"] = df["fighter_vitorias"] - df["opponent_vitorias"]
    f["d_derrotas"] = df["fighter_derrotas"] - df["opponent_derrotas"]
    f["d_lutas_totais"] = df["fighter_lutas_totais"] - df["opponent_lutas_totais"]

    # Taxas de vitória por método (nocaute / finalização / decisão)
    for lado in ("fighter", "opponent"):
        v = df[f"{lado}_vitorias"]
        ko = df[f"{lado}_ko"]
        sub = df[f"{lado}_sub"]
        dec = (v - ko - sub).clip(lower=0)
        df[f"{lado}_taxa_ko"] = _taxa(ko, v)
        df[f"{lado}_taxa_sub"] = _taxa(sub, v)
        df[f"{lado}_taxa_dec"] = _taxa(dec, v)
    f["d_taxa_ko"] = df["fighter_taxa_ko"] - df["opponent_taxa_ko"]
    f["d_taxa_sub"] = df["fighter_taxa_sub"] - df["opponent_taxa_sub"]
    f["d_taxa_dec"] = df["fighter_taxa_dec"] - df["opponent_taxa_dec"]

    # Estilo de luta: diferença de indicadores por estilo (antissimétrica)
    for e in config.ESTILOS:
        ind_f = (df["fighter_stance"] == e).astype(int)
        ind_o = (df["opponent_stance"] == e).astype(int)
        f[f"d_stance_{e}"] = ind_f - ind_o
    # Indicador simétrico: os lutadores têm estilos diferentes?
    f[config.FEATURE_ESTILOS_DIF] = (
        df["fighter_stance"] != df["opponent_stance"]
    ).astype(int)

    return f


def espelhar(df):
    """Duplica cada luta com os cantos trocados (features negadas, rótulo invertido)."""
    espelho = df.copy()
    cols_negar = config.FEATURES_DIFF + config.FEATURES_STANCE
    espelho[cols_negar] = -espelho[cols_negar]
    espelho[config.ALVO] = 1 - espelho[config.ALVO]
    # 'estilos_diferentes' é simétrico, permanece igual.
    return pd.concat([df, espelho], ignore_index=True)


def gerar():
    config.garantir_diretorios()
    df = pd.read_csv(config.CSV_PROC, parse_dates=["data"])
    feats = construir_features(df).sort_values("data").reset_index(drop=True)

    # Split temporal: treino = lutas antigas, teste = lutas recentes
    n = len(feats)
    corte = int(n * (1 - config.FRAC_TESTE))
    treino = feats.iloc[:corte].copy()
    teste = feats.iloc[corte:].copy()

    data_corte = teste["data"].min()
    print(f"Total: {n} lutas | corte temporal em {data_corte.date()}")
    print(f"Treino: {len(treino)} lutas ({treino['data'].dt.year.min()}-{treino['data'].dt.year.max()})")
    print(f"Teste : {len(teste)} lutas ({teste['data'].dt.year.min()}-{teste['data'].dt.year.max()})")

    # Espelhamento apenas no treino (data augmentation antissimétrica)
    treino_esp = espelhar(treino)
    print(f"Treino após espelhamento: {len(treino_esp)} exemplos "
          f"| classe positiva: {treino_esp[config.ALVO].mean():.3f}")
    print(f"Teste classe positiva: {teste[config.ALVO].mean():.3f}")

    treino_esp.to_csv(config.CSV_TREINO, index=False)
    teste.to_csv(config.CSV_TESTE, index=False)
    print(f"\nSalvos:\n  {config.CSV_TREINO}\n  {config.CSV_TESTE}")


if __name__ == "__main__":
    gerar()
