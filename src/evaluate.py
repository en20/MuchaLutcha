"""Etapa 5 do pipeline: avaliação dos modelos no conjunto de teste.

Calcula as métricas de desempenho (acurácia, precisão, revocação, F1, AUC-ROC e
log-loss) no conjunto de teste temporal, compara todos os modelos com os
baselines e gera as figuras do artigo (curvas ROC, matrizes de confusão,
importância de atributos e curva de calibração). Exporta tabelas em CSV e LaTeX.
"""
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)

import config
import viz

warnings.filterwarnings("ignore")

NOMES = {
    "dummy": "Baseline (classe majoritária)",
    "regra_cartel": "Baseline (melhor cartel)",
    "logreg": "Regressão Logística",
    "rf": "Random Forest",
    "xgboost": "XGBoost",
    "svm": "SVM (RBF)",
    "mlp": "Rede Neural (MLP)",
}
MODELOS_ML = ["logreg", "rf", "xgboost", "svm", "mlp"]


def metricas(y, y_pred, y_score):
    d = {
        "Acurácia": accuracy_score(y, y_pred),
        "Precisão": precision_score(y, y_pred, zero_division=0),
        "Revocação": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
    }
    if y_score is not None:
        d["AUC-ROC"] = roc_auc_score(y, y_score)
        d["Log-loss"] = log_loss(y, np.clip(y_score, 1e-6, 1 - 1e-6))
    else:
        d["AUC-ROC"], d["Log-loss"] = np.nan, np.nan
    return d


def carregar_dados():
    teste = pd.read_csv(config.CSV_TESTE)
    X = teste[config.FEATURES].values
    y = teste[config.ALVO].values
    return teste, X, y


def avaliar_todos(teste, X, y):
    """Retorna (tabela de métricas, dict de probabilidades por modelo)."""
    linhas, probs = [], {}

    # Baseline 1: classe majoritária
    dummy = joblib.load(config.DIR_MODELOS / "dummy.joblib")
    linhas.append({"Modelo": NOMES["dummy"], **metricas(y, dummy.predict(X), None)})

    # Baseline 2: prever o lutador com melhor cartel (sinal de d_vitorias)
    d_vit = teste["d_vitorias"].values
    pred_cartel = (d_vit > 0).astype(int)
    score_cartel = 1 / (1 + np.exp(-d_vit / 5.0))  # escala suave só para AUC/log-loss
    linhas.append({"Modelo": NOMES["regra_cartel"], **metricas(y, pred_cartel, score_cartel)})

    # Modelos de ML
    for chave in MODELOS_ML:
        modelo = joblib.load(config.DIR_MODELOS / f"{chave}.joblib")
        p = modelo.predict_proba(X)[:, 1]
        probs[chave] = p
        linhas.append({"Modelo": NOMES[chave], **metricas(y, (p >= 0.5).astype(int), p)})

    tabela = pd.DataFrame(linhas)
    return tabela, probs


def exportar_tabela(tabela):
    cols = ["Modelo", "Acurácia", "Precisão", "Revocação", "F1", "AUC-ROC", "Log-loss"]
    tabela = tabela[cols].copy()
    tabela.to_csv(config.DIR_TABELAS / "metricas_teste.csv", index=False)

    fmt = tabela.copy()
    for c in cols[1:]:
        fmt[c] = fmt[c].map(lambda v: "--" if pd.isna(v) else f"{v:.3f}")

    # Destaca em negrito o melhor valor de cada métrica (menor para log-loss)
    melhor = {}
    for c in cols[1:]:
        vals = pd.to_numeric(tabela[c], errors="coerce")
        melhor[c] = (vals.idxmin() if c == "Log-loss" else vals.idxmax())

    linhas = [r"\begin{table}[t]", r"\centering",
              r"\caption{Desempenho dos modelos no conjunto de teste temporal "
              r"(lutas de 2021--2024). Em negrito, o melhor valor de cada métrica.}",
              r"\label{tab:metricas}",
              r"\begin{tabular}{lcccccc}", r"\toprule",
              "Modelo & Acurácia & Precisão & Revocação & F1 & AUC-ROC & Log-loss \\\\",
              r"\midrule"]
    for i, row in fmt.iterrows():
        celulas = [str(row["Modelo"])]
        for c in cols[1:]:
            txt = row[c]
            if i == melhor[c] and txt != "--":
                txt = r"\textbf{" + txt + "}"
            celulas.append(txt)
        linhas.append(" & ".join(celulas) + r" \\")
    linhas += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (config.DIR_TABELAS / "metricas_teste.tex").write_text("\n".join(linhas), encoding="utf-8")
    print("Tabela de métricas:")
    print(fmt.to_string(index=False))


def fig_roc(y, probs):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for chave in MODELOS_ML:
        fpr, tpr, _ = roc_curve(y, probs[chave])
        auc = roc_auc_score(y, probs[chave])
        ax.plot(fpr, tpr, lw=1.8, label=f"{NOMES[chave]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title("Curvas ROC no conjunto de teste")
    ax.legend(loc="lower right", fontsize=8)
    viz.salvar(fig, "roc")


def fig_confusao(y, probs):
    fig, axes = plt.subplots(1, len(MODELOS_ML), figsize=(15, 3.1))
    import seaborn as sns
    for ax, chave in zip(axes, MODELOS_ML):
        cm = confusion_matrix(y, (probs[chave] >= 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Derrota", "Vitória"],
                    yticklabels=["Derrota", "Vitória"])
        ax.set_title(NOMES[chave], fontsize=9)
        ax.set_xlabel("Previsto")
        if chave == MODELOS_ML[0]:
            ax.set_ylabel("Real")
    fig.suptitle("Matrizes de confusão no conjunto de teste")
    fig.tight_layout()
    viz.salvar(fig, "confusao")


def fig_importancia(X, y):
    """Coeficientes da Regressão Logística + importância por permutação."""
    logreg = joblib.load(config.DIR_MODELOS / "logreg.joblib")
    coef = logreg.named_steps["clf"].coef_[0]

    perm = permutation_importance(logreg, X, y, n_repeats=20,
                                  random_state=config.SEMENTE, scoring="roc_auc")
    imp = perm.importances_mean

    ordem = np.argsort(np.abs(coef))
    nomes = np.array(config.FEATURES)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(nomes[ordem], coef[ordem], color="#4c72b0")
    axes[0].axvline(0, color="gray", lw=1)
    axes[0].set_title("Coeficientes da Regressão Logística")
    axes[0].set_xlabel("Peso (dados padronizados)")

    ordem2 = np.argsort(imp)
    axes[1].barh(nomes[ordem2], imp[ordem2], color="#55a868")
    axes[1].set_title("Importância por permutação (AUC)")
    axes[1].set_xlabel("Queda média na AUC")
    fig.tight_layout()
    viz.salvar(fig, "importancia")


def fig_calibracao(y, probs, chave="logreg"):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    frac_pos, media_prev = calibration_curve(y, probs[chave], n_bins=10, strategy="quantile")
    ax.plot(media_prev, frac_pos, marker="o", color="#4c72b0", label=NOMES[chave])
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfeitamente calibrado")
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Fração observada de vitórias")
    ax.set_title("Curva de calibração")
    ax.legend(loc="upper left", fontsize=9)
    viz.salvar(fig, "calibracao")


def main():
    config.garantir_diretorios()
    teste, X, y = carregar_dados()
    print(f"Teste: {len(y)} lutas | classe positiva: {y.mean():.3f}\n")

    tabela, probs = avaliar_todos(teste, X, y)
    exportar_tabela(tabela)

    print("\nGerando figuras...")
    fig_roc(y, probs)
    fig_confusao(y, probs)
    fig_importancia(X, y)
    fig_calibracao(y, probs)
    print("Avaliação concluída.")


if __name__ == "__main__":
    main()
