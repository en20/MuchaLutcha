"""Etapa 4 do pipeline: treino e ajuste de hiperparâmetros dos modelos.

Treina cinco modelos de aprendizagem supervisionada para o problema de
classificação binária (vitória do lutador). Cada modelo é encapsulado em um
``Pipeline`` (com padronização quando aplicável) e tem seus hiperparâmetros
ajustados por ``GridSearchCV`` com validação cruzada estratificada, usando
**apenas o conjunto de treino**. Os melhores estimadores são salvos em
``results/models`` e um resumo da seleção em ``results/tables``.
"""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import config

warnings.filterwarnings("ignore")


def definir_modelos():
    """Retorna {chave: (pipeline, grade_de_hiperparametros)} para cada modelo."""
    esc = ("esc", StandardScaler())  # padronização para modelos sensíveis à escala
    modelos = {}

    modelos["logreg"] = (
        Pipeline([esc, ("clf", LogisticRegression(max_iter=2000))]),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    )

    modelos["rf"] = (
        Pipeline([("clf", RandomForestClassifier(
            n_estimators=400, random_state=config.SEMENTE, n_jobs=-1))]),
        {"clf__max_depth": [None, 6, 12],
         "clf__min_samples_leaf": [1, 5, 20],
         "clf__max_features": ["sqrt", 0.5]},
    )

    modelos["xgboost"] = (
        Pipeline([("clf", XGBClassifier(
            n_estimators=400, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=config.SEMENTE, n_jobs=-1))]),
        {"clf__max_depth": [3, 5],
         "clf__learning_rate": [0.03, 0.1],
         "clf__reg_lambda": [1.0, 5.0]},
    )

    modelos["svm"] = (
        Pipeline([esc, ("clf", SVC(kernel="rbf", probability=True,
                                   random_state=config.SEMENTE))]),
        {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", 0.1]},
    )

    modelos["mlp"] = (
        Pipeline([esc, ("clf", MLPClassifier(
            max_iter=800, early_stopping=True, random_state=config.SEMENTE))]),
        {"clf__hidden_layer_sizes": [(32,), (64, 32)],
         "clf__alpha": [1e-4, 1e-3]},
    )

    return modelos


def treinar():
    config.garantir_diretorios()
    treino = pd.read_csv(config.CSV_TREINO)
    X = treino[config.FEATURES].values
    y = treino[config.ALVO].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEMENTE)
    resumo = []

    # Baseline trivial (classe mais frequente) treinado como referência mínima
    dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
    joblib.dump(dummy, config.DIR_MODELOS / "dummy.joblib")

    for chave, (pipe, grade) in definir_modelos().items():
        t0 = time.time()
        busca = GridSearchCV(pipe, grade, scoring="roc_auc", cv=cv,
                             n_jobs=-1, refit=True)
        busca.fit(X, y)
        dt = time.time() - t0
        joblib.dump(busca.best_estimator_, config.DIR_MODELOS / f"{chave}.joblib")
        resumo.append({
            "modelo": chave,
            "auc_cv": round(busca.best_score_, 4),
            "tempo_s": round(dt, 1),
            "melhores_params": json.dumps(busca.best_params_),
        })
        print(f"{chave:8s} | AUC(cv)={busca.best_score_:.4f} | {dt:5.1f}s | {busca.best_params_}")

    df_resumo = pd.DataFrame(resumo).sort_values("auc_cv", ascending=False)
    df_resumo.to_csv(config.DIR_TABELAS / "selecao_modelos.csv", index=False)
    print(f"\nResumo salvo em {config.DIR_TABELAS / 'selecao_modelos.csv'}")
    print(df_resumo.to_string(index=False))


if __name__ == "__main__":
    treinar()
