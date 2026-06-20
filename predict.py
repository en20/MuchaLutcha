"""
MuchaLutcha — UFC Fight Predictor
Interactive CLI to predict UFC fight outcomes using the trained models.

Usage:
    python predict.py                      # interactive mode (prompt fighter names)
    python predict.py --demo               # run 10 real historical fight examples
    python predict.py --fight "Jon Jones" "Stipe Miocic"   # direct prediction
"""
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

# Add src/ to path so config is importable from the project root
sys.path.insert(0, "src")
import config

MODELOS_DISPONIVEIS = ["rf", "logreg", "xgboost", "svm", "mlp"]
NOMES = {
    "rf": "Random Forest",
    "logreg": "Regressão Logística",
    "xgboost": "XGBoost",
    "svm": "SVM (RBF)",
    "mlp": "Rede Neural (MLP)",
}
ESTILOS_VALIDOS = ["orthodox", "southpaw", "switch", "other"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def carregar_lutadores():
    """Load fighter stats for auto-lookup."""
    try:
        s = pd.read_csv(config.CSV_LUTADORES)
        s = s.drop_duplicates(subset="name", keep="first")
        s["altura_cm"] = s["height"].apply(_alt_to_cm)
        s["envergadura_cm"] = s["reach"].apply(_env_to_cm)
        s["stance_norm"] = s["stance"].apply(_norm_stance)
        return s.set_index("name")
    except FileNotFoundError:
        return None


def _alt_to_cm(valor):
    import re
    if not isinstance(valor, str):
        return np.nan
    m = re.match(r"\s*(\d+)\s*'\s*(\d+)", valor)
    if not m:
        return np.nan
    return (int(m.group(1)) * 12 + int(m.group(2))) * 2.54


def _env_to_cm(valor):
    import re
    if not isinstance(valor, str):
        return np.nan
    m = re.match(r"\s*(\d+(?:\.\d+)?)", valor)
    if not m:
        return np.nan
    return float(m.group(1)) * 2.54


def _norm_stance(valor):
    if not isinstance(valor, str):
        return "other"
    v = valor.strip().lower()
    return v if v in ("orthodox", "southpaw", "switch") else "other"


def buscar_lutador(nome: str, df_stats, df_fights) -> dict | None:
    """Try to find fighter stats by name (exact match, then fuzzy)."""
    if df_stats is not None and nome in df_stats.index:
        row = df_stats.loc[nome]
        altura = float(row["altura_cm"]) if not pd.isna(row["altura_cm"]) else 177.8
        env = float(row["envergadura_cm"]) if not pd.isna(row["envergadura_cm"]) else 182.9
        stance = row["stance_norm"]
        # Latest fight record for wins/losses/KO/sub
        rec = _record_from_fights(nome, df_fights)
        return {"nome": nome, "altura": altura, "envergadura": env, "stance": stance, **rec}

    # Fuzzy: case-insensitive substring
    if df_stats is not None:
        nome_lower = nome.lower()
        matches = [n for n in df_stats.index if nome_lower in n.lower()]
        if len(matches) == 1:
            return buscar_lutador(matches[0], df_stats, df_fights)
        if len(matches) > 1:
            print(f"  Múltiplos matches para '{nome}': {matches[:5]}")
            return None
    return None


def _record_from_fights(nome: str, df_fights) -> dict:
    """Extract latest fight record stats for a fighter."""
    if df_fights is None:
        return {"idade": 30.0, "vitorias": 0, "derrotas": 0, "lutas_totais": 0,
                "ko": 0, "sub": 0}

    as_fighter = df_fights[df_fights["fighter"] == nome].sort_values("data")
    if len(as_fighter):
        last = as_fighter.iloc[-1]
        return {
            "idade": float(last["fighter_idade"]) if not pd.isna(last["fighter_idade"]) else 30.0,
            "vitorias": int(last["fighter_vitorias"]),
            "derrotas": int(last["fighter_derrotas"]),
            "lutas_totais": int(last["fighter_lutas_totais"]),
            "ko": int(last["fighter_ko"]),
            "sub": int(last["fighter_sub"]),
        }
    return {"idade": 30.0, "vitorias": 0, "derrotas": 0, "lutas_totais": 0, "ko": 0, "sub": 0}


def _taxa(num, den):
    return num / den if den > 0 else 0.0


def construir_vetor(f: dict, o: dict) -> np.ndarray:
    """Build the 14-dimensional feature vector from two fighter dicts."""
    # Taxas de vitória por método
    f_taxa_ko = _taxa(f["ko"], f["vitorias"])
    f_taxa_sub = _taxa(f["sub"], f["vitorias"])
    f_taxa_dec = _taxa(max(f["vitorias"] - f["ko"] - f["sub"], 0), f["vitorias"])
    o_taxa_ko = _taxa(o["ko"], o["vitorias"])
    o_taxa_sub = _taxa(o["sub"], o["vitorias"])
    o_taxa_dec = _taxa(max(o["vitorias"] - o["ko"] - o["sub"], 0), o["vitorias"])

    row = {
        "d_idade": f["idade"] - o["idade"],
        "d_altura": f["altura"] - o["altura"],
        "d_envergadura": f["envergadura"] - o["envergadura"],
        "d_vitorias": f["vitorias"] - o["vitorias"],
        "d_derrotas": f["derrotas"] - o["derrotas"],
        "d_lutas_totais": f["lutas_totais"] - o["lutas_totais"],
        "d_taxa_ko": f_taxa_ko - o_taxa_ko,
        "d_taxa_sub": f_taxa_sub - o_taxa_sub,
        "d_taxa_dec": f_taxa_dec - o_taxa_dec,
    }
    for e in config.ESTILOS:
        row[f"d_stance_{e}"] = (1 if f["stance"] == e else 0) - (1 if o["stance"] == e else 0)
    row["estilos_diferentes"] = int(f["stance"] != o["stance"])

    return np.array([row[feat] for feat in config.FEATURES], dtype=float)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def carregar_modelos():
    modelos = {}
    for chave in MODELOS_DISPONIVEIS:
        path = config.DIR_MODELOS / f"{chave}.joblib"
        if path.exists():
            modelos[chave] = joblib.load(path)
    return modelos


def prever(f: dict, o: dict, modelos: dict) -> dict:
    """Return dict of model_key -> win_probability for fighter f."""
    X = construir_vetor(f, o).reshape(1, -1)
    return {chave: modelo.predict_proba(X)[0, 1] for chave, modelo in modelos.items()}


def ensemble_prob(probs: dict) -> float:
    return float(np.mean(list(probs.values())))


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

LARGURA = 68


def _barra(prob: float, largura: int = 40) -> str:
    n = round(prob * largura)
    return "█" * n + "░" * (largura - n)


def _cabecalho():
    print("=" * LARGURA)
    print("  🥊  MuchaLutcha — Previsão de Lutas do UFC  🥊")
    print("=" * LARGURA)


def _separador():
    print("-" * LARGURA)


def exibir_resultado(f: dict, o: dict, probs: dict, resultado_real: str | None = None):
    """Pretty-print prediction results for a matchup."""
    prob_f = ensemble_prob(probs)
    prob_o = 1.0 - prob_f

    _separador()
    print(f"  {f['nome']:>28}  vs  {o['nome']:<28}")
    _separador()

    # Fighter stats side-by-side
    print(f"  {'':>28}       {'':>28}")
    print(f"  {'Vitórias: ' + str(f['vitorias']):>28}  |  {'Vitórias: ' + str(o['vitorias']):<28}")
    print(f"  {'Derrotas: ' + str(f['derrotas']):>28}  |  {'Derrotas: ' + str(o['derrotas']):<28}")
    print(f"  {'Altura: ' + str(round(f['altura'], 0)) + ' cm':>28}  |  {'Altura: ' + str(round(o['altura'], 0)) + ' cm':<28}")
    print(f"  {'Envergadura: ' + str(round(f['envergadura'], 0)) + ' cm':>28}  |  {'Envergadura: ' + str(round(o['envergadura'], 0)) + ' cm':<28}")
    print(f"  {'Stance: ' + f['stance']:>28}  |  {'Stance: ' + o['stance']:<28}")
    _separador()

    # Per-model predictions
    print(f"  {'Modelo':<24} {'P(' + f['nome'].split()[0] + ')':<10} {'P(' + o['nome'].split()[0] + ')':<10}")
    print(f"  {'─'*24} {'─'*10} {'─'*10}")
    for chave, p in probs.items():
        bar = "◄" if p >= 0.5 else " "
        bar2 = "►" if p < 0.5 else " "
        print(f"  {NOMES[chave]:<24} {p:.1%}{bar:<6} {(1-p):.1%}{bar2}")
    _separador()

    # Ensemble result
    winner = f["nome"] if prob_f >= 0.5 else o["nome"]
    conf = max(prob_f, prob_o)
    print(f"\n  PREVISÃO: {winner}")
    print(f"  Confiança: {conf:.1%}")
    print()
    print(f"  {f['nome'].split()[0][:12]:<14} {_barra(prob_f, 36)} {o['nome'].split()[0][:12]}")
    print(f"  {(f'{prob_f:.1%}'):<14} {'':36} {prob_o:.1%}")
    print()

    if resultado_real is not None:
        acertou = "✓ CORRETO" if winner == resultado_real else "✗ ERRADO"
        print(f"  Resultado real: {resultado_real}  →  {acertou}")
    print()


# ---------------------------------------------------------------------------
# Demo fights (real historical results)
# ---------------------------------------------------------------------------

DEMO_FIGHTS = [
    # (fighter, opponent, winner, date)
    ("Jon Jones", "Stipe Miocic", "Jon Jones", "2023-11-11"),
    ("Islam Makhachev", "Dustin Poirier", "Islam Makhachev", "2024-06-01"),
    ("Alexandre Pantoja", "Kai Asakura", "Alexandre Pantoja", "2024-12-07"),
    ("Conor McGregor", "Khabib Nurmagomedov", "Khabib Nurmagomedov", "2018-10-06"),
    ("Colby Covington", "Joaquin Buckley", "Joaquin Buckley", "2024-12-14"),
    ("Ciryl Gane", "Alexander Volkov", "Ciryl Gane", "2024-12-07"),
    ("Ian Machado Garry", "Shavkat Rakhmonov", "Shavkat Rakhmonov", "2024-12-07"),
    ("Dustin Poirier", "Justin Gaethje", "Justin Gaethje", "2024-05-18"),
    ("Alex Pereira", "Jiri Prochazka", "Alex Pereira", "2024-06-29"),
    ("Sean O'Malley", "Merab Dvalishvili", "Merab Dvalishvili", "2024-09-14"),
]


def rodar_demo(modelos, df_stats, df_fights):
    _cabecalho()
    print("  Modo DEMO — 10 lutas reais com resultados conhecidos")
    print()

    acertos = 0
    for nome_f, nome_o, vencedor_real, data in DEMO_FIGHTS:
        f = buscar_lutador(nome_f, df_stats, df_fights)
        o = buscar_lutador(nome_o, df_stats, df_fights)

        if f is None:
            print(f"  [!] '{nome_f}' não encontrado no dataset — pulando.")
            continue
        if o is None:
            print(f"  [!] '{nome_o}' não encontrado no dataset — pulando.")
            continue

        probs = prever(f, o, modelos)
        prob_f = ensemble_prob(probs)
        previsto = f["nome"] if prob_f >= 0.5 else o["nome"]
        exibir_resultado(f, o, probs, resultado_real=vencedor_real)
        if previsto == vencedor_real:
            acertos += 1

    _separador()
    total = len(DEMO_FIGHTS)
    print(f"  RESUMO DEMO: {acertos}/{total} previsões corretas ({acertos/total:.0%})")
    print("=" * LARGURA)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def _pedir_stats_manual(nome: str) -> dict:
    """Ask user for fighter stats manually when not found in DB."""
    print(f"\n  Dados de '{nome}' não encontrados. Insira manualmente:")
    idade = float(input("  Idade (anos): ") or "30")
    altura = float(input("  Altura (cm): ") or "177.8")
    env = float(input("  Envergadura (cm): ") or "182.9")
    print(f"  Stance (orthodox/southpaw/switch/other): ", end="")
    stance = input().strip().lower() or "orthodox"
    if stance not in ESTILOS_VALIDOS:
        stance = "other"
    vitorias = int(input("  Vitórias: ") or "0")
    derrotas = int(input("  Derrotas: ") or "0")
    ko = int(input("  KO wins: ") or "0")
    sub = int(input("  SUB wins: ") or "0")
    return {
        "nome": nome, "idade": idade, "altura": altura, "envergadura": env,
        "stance": stance, "vitorias": vitorias, "derrotas": derrotas,
        "lutas_totais": vitorias + derrotas, "ko": ko, "sub": sub,
    }


def modo_interativo(modelos, df_stats, df_fights):
    _cabecalho()
    print("  Digite os nomes dos lutadores para ver a previsão.")
    print("  Use exatamente como no dataset (ex: 'Jon Jones').")
    print("  Ctrl+C para sair.")
    print()

    while True:
        try:
            print()
            nome_f = input("  Lutador: ").strip()
            if not nome_f:
                continue
            nome_o = input("  Oponente: ").strip()
            if not nome_o:
                continue

            f = buscar_lutador(nome_f, df_stats, df_fights) or _pedir_stats_manual(nome_f)
            o = buscar_lutador(nome_o, df_stats, df_fights) or _pedir_stats_manual(nome_o)

            probs = prever(f, o, modelos)
            exibir_resultado(f, o, probs)

        except KeyboardInterrupt:
            print("\n\n  Saindo. Valeu! 🥊")
            break


# ---------------------------------------------------------------------------
# Direct prediction mode
# ---------------------------------------------------------------------------

def modo_direto(nome_f: str, nome_o: str, modelos, df_stats, df_fights):
    _cabecalho()
    f = buscar_lutador(nome_f, df_stats, df_fights) or _pedir_stats_manual(nome_f)
    o = buscar_lutador(nome_o, df_stats, df_fights) or _pedir_stats_manual(nome_o)
    probs = prever(f, o, modelos)
    exibir_resultado(f, o, probs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UFC Fight Predictor")
    parser.add_argument("--demo", action="store_true", help="Run 10 real historical fight examples")
    parser.add_argument("--fight", nargs=2, metavar=("FIGHTER", "OPPONENT"),
                        help="Predict a specific matchup")
    args = parser.parse_args()

    modelos = carregar_modelos()
    if not modelos:
        print("Nenhum modelo encontrado em results/models/. Rode train.py primeiro.")
        sys.exit(1)
    print(f"  Modelos carregados: {', '.join(modelos.keys())}")

    df_fights = pd.read_csv(config.CSV_PROC, parse_dates=["data"]) if config.CSV_PROC.exists() else None
    df_stats = carregar_lutadores()

    if args.demo:
        rodar_demo(modelos, df_stats, df_fights)
    elif args.fight:
        modo_direto(args.fight[0], args.fight[1], modelos, df_stats, df_fights)
    else:
        modo_interativo(modelos, df_stats, df_fights)


if __name__ == "__main__":
    main()
