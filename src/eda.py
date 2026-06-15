"""Etapa 3 do pipeline: análise exploratória dos dados (EDA).

Gera figuras descritivas usadas na seção de Fundamentação Teórica/Experimentos
do artigo: volume de lutas ao longo do tempo, distribuição dos atributos,
correlação entre features e relação entre features diferenciais e a vitória.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import viz


def fig_lutas_por_ano(df):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    contagem = df["data"].dt.year.value_counts().sort_index()
    ax.bar(contagem.index, contagem.values, color="#4c72b0")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Nº de lutas")
    ax.set_title("Volume de lutas do UFC por ano")
    viz.salvar(fig, "eda_lutas_por_ano")


def fig_distribuicao_atributos(df):
    """Distribuição de idade, altura e envergadura dos lutadores."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    specs = [
        ("fighter_idade", "Idade (anos)"),
        ("fighter_altura", "Altura (cm)"),
        ("fighter_envergadura", "Envergadura (cm)"),
    ]
    for ax, (col, rotulo) in zip(axes, specs):
        ax.hist(df[col].dropna(), bins=30, color="#55a868", edgecolor="white")
        ax.set_xlabel(rotulo)
        ax.set_ylabel("Frequência")
    fig.suptitle("Distribuição dos atributos físicos dos lutadores")
    fig.tight_layout()
    viz.salvar(fig, "eda_distribuicao_atributos")


def fig_estilos(df):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ordem = ["orthodox", "southpaw", "switch", "other"]
    cont = df["fighter_stance"].value_counts().reindex(ordem).fillna(0)
    ax.bar(cont.index, cont.values, color="#c44e52")
    ax.set_xlabel("Estilo de luta (stance)")
    ax.set_ylabel("Nº de lutadores-luta")
    ax.set_title("Distribuição dos estilos de luta")
    viz.salvar(fig, "eda_estilos")


def fig_correlacao(treino):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    corr = treino[config.FEATURES].corr()
    import seaborn as sns
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.7}, linewidths=0.3)
    ax.set_title("Correlação entre as features")
    viz.salvar(fig, "eda_correlacao")


def fig_winrate_vs_feature(treino):
    """Taxa de vitória do lutador em função de features diferenciais (binadas)."""
    specs = [("d_vitorias", "Diferença de vitórias"),
             ("d_idade", "Diferença de idade (anos)"),
             ("d_envergadura", "Diferença de envergadura (cm)"),
             ("d_taxa_ko", "Diferença de taxa de nocaute")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (col, rotulo) in zip(axes.ravel(), specs):
        x = treino[col]
        bins = np.quantile(x, np.linspace(0, 1, 11))
        bins = np.unique(bins)
        cat = pd.cut(x, bins=bins, include_lowest=True)
        taxa = treino.groupby(cat, observed=True)[config.ALVO].mean()
        centros = [iv.mid for iv in taxa.index]
        ax.plot(centros, taxa.values, marker="o", color="#4c72b0")
        ax.axhline(0.5, ls="--", color="gray", lw=1)
        ax.set_xlabel(rotulo)
        ax.set_ylabel("Taxa de vitória")
        ax.set_ylim(0, 1)
    fig.suptitle("Relação entre features diferenciais e a probabilidade de vitória")
    fig.tight_layout()
    viz.salvar(fig, "eda_winrate_vs_feature")


def main():
    config.garantir_diretorios()
    df = pd.read_csv(config.CSV_PROC, parse_dates=["data"])
    treino = pd.read_csv(config.CSV_TREINO, parse_dates=["data"])
    print("Gerando figuras de EDA...")
    fig_lutas_por_ano(df)
    fig_distribuicao_atributos(df)
    fig_estilos(df)
    fig_correlacao(treino)
    fig_winrate_vs_feature(treino)
    print("EDA concluída.")


if __name__ == "__main__":
    main()
