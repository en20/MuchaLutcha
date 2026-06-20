"""
MuchaLutcha Web App — Flask backend
Serves the UFC MatchMaker UI and provides three JSON endpoints:
  GET  /api/rankings          → scraped UFC rankings by division
  GET  /api/fighter/<slug>    → fighter stats from local dataset
  POST /api/predict           → ML ensemble prediction
"""
import sys
import time
import json
import threading
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent / "src"))
import config

app = Flask(__name__, static_folder="web", static_url_path="")
CORS(app)

# ---------------------------------------------------------------------------
# ML model loading
# ---------------------------------------------------------------------------
_models = {}
_fights_df = None
_stats_df = None


def load_models():
    global _models, _fights_df, _stats_df
    for key in ["rf", "logreg", "xgboost", "svm", "mlp"]:
        p = config.DIR_MODELOS / f"{key}.joblib"
        if p.exists():
            _models[key] = joblib.load(p)
    if config.CSV_PROC.exists():
        _fights_df = pd.read_csv(config.CSV_PROC, parse_dates=["data"])
    if config.CSV_LUTADORES.exists():
        import re
        s = pd.read_csv(config.CSV_LUTADORES)
        s = s.drop_duplicates(subset="name", keep="first")

        def _alt(v):
            if not isinstance(v, str): return None
            m = re.match(r"\s*(\d+)\s*'\s*(\d+)", v)
            return (int(m.group(1)) * 12 + int(m.group(2))) * 2.54 if m else None

        def _env(v):
            if not isinstance(v, str): return None
            m = re.match(r"\s*(\d+(?:\.\d+)?)", v)
            return float(m.group(1)) * 2.54 if m else None

        def _stance(v):
            if not isinstance(v, str): return "other"
            v = v.strip().lower()
            return v if v in ("orthodox", "southpaw", "switch") else "other"

        s["altura_cm"] = s["height"].apply(_alt)
        s["envergadura_cm"] = s["reach"].apply(_env)
        s["stance_norm"] = s["stance"].apply(_stance)
        _stats_df = s.set_index("name")


load_models()

# ---------------------------------------------------------------------------
# UFC rankings scraper (cached 10 min)
# ---------------------------------------------------------------------------

DIVISION_MAP = {
    # Portuguese names as they actually appear on ufc.com/rankings (pt-BR)
    "Peso-mosca": "flyweight",
    "Peso-galo": "bantamweight",
    "Peso-pena": "featherweight",
    "Peso-leve": "lightweight",
    "Peso Meio-Médio": "welterweight",
    "Peso-médio": "middleweight",
    "Peso meio-pesado": "light-heavyweight",
    "Peso-pesado": "heavyweight",
    "Peso-palha feminino": "womens-strawweight",
    "Peso-mosca feminino": "womens-flyweight",
    "Peso-galo feminino": "womens-bantamweight",
    # English fallbacks
    "Flyweight": "flyweight",
    "Bantamweight": "bantamweight",
    "Featherweight": "featherweight",
    "Lightweight": "lightweight",
    "Welterweight": "welterweight",
    "Middleweight": "middleweight",
    "Light Heavyweight": "light-heavyweight",
    "Heavyweight": "heavyweight",
    "Women's Strawweight": "womens-strawweight",
    "Women's Flyweight": "womens-flyweight",
    "Women's Bantamweight": "womens-bantamweight",
}

_rankings_cache = {"data": None, "ts": 0}
_rankings_lock = threading.Lock()


def scrape_rankings():
    url = "https://www.ufc.com/rankings"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    result = {}
    for group in soup.select(".view-grouping"):
        header = group.select_one(".view-grouping-header")
        if not header:
            continue
        division_pt = header.get_text(strip=True)
        div_key = DIVISION_MAP.get(division_pt)
        if not div_key:
            continue  # skip P4P

        fighters = []

        # Champion
        champ_el = group.select_one(".rankings--athlete--champion h5 a")
        if champ_el:
            fighters.append({
                "rank": "C",
                "name": champ_el.get_text(strip=True),
                "slug": champ_el["href"].replace("/athlete/", ""),
                "is_champ": True,
            })

        # Ranked fighters
        for row in group.select("tr"):
            rank_td = row.select_one(".views-field-weight-class-rank")
            name_td = row.select_one(".views-field-title a")
            if not rank_td or not name_td:
                continue
            rank_text = rank_td.get_text(strip=True)
            try:
                rank_num = int(rank_text)
            except ValueError:
                continue
            fighters.append({
                "rank": rank_num,
                "name": name_td.get_text(strip=True),
                "slug": name_td["href"].replace("/athlete/", ""),
                "is_champ": False,
            })

        result[div_key] = fighters

    return result


def get_rankings():
    with _rankings_lock:
        now = time.time()
        if _rankings_cache["data"] is None or now - _rankings_cache["ts"] > 600:
            try:
                _rankings_cache["data"] = scrape_rankings()
                _rankings_cache["ts"] = now
            except Exception as e:
                if _rankings_cache["data"] is None:
                    raise
                app.logger.warning(f"Rankings scrape failed, using cache: {e}")
        return _rankings_cache["data"]


# ---------------------------------------------------------------------------
# Fighter stats lookup from local dataset
# ---------------------------------------------------------------------------

def fighter_record(name):
    """Return latest win/loss/ko/sub counts for a fighter name."""
    if _fights_df is None:
        return {"wins": 0, "losses": 0, "ko": 0, "sub": 0, "age": 30,
                "height": 177.8, "reach": 182.9, "stance": "orthodox"}

    mask = _fights_df["fighter"] == name
    rows = _fights_df[mask].sort_values("data")
    if rows.empty:
        # try case-insensitive
        mask2 = _fights_df["fighter"].str.lower() == name.lower()
        rows = _fights_df[mask2].sort_values("data")

    if rows.empty:
        return None

    last = rows.iloc[-1]

    def _f(col, default=0.0):
        v = last.get(col, default)
        return float(v) if not pd.isna(v) else default

    return {
        "wins":   int(_f("fighter_vitorias")),
        "losses": int(_f("fighter_derrotas")),
        "ko":     int(_f("fighter_ko")),
        "sub":    int(_f("fighter_sub")),
        "ko_losses": int(_f("fighter_ko_losses")),
        "age":    _f("fighter_idade", 30.0),
        "height": round(_f("fighter_altura", 177.8), 1),
        "reach":  round(_f("fighter_envergadura", 182.9), 1),
        "stance": str(last.get("fighter_stance", "orthodox")),
        "L5Y_winrate": _f("fighter_L5Y_winrate", 0.5),
        "L2Y_winrate": _f("fighter_L2Y_winrate", 0.5),
        "sig_strikes_landed":   _f("fighter_sig_strikes_landed"),
        "sig_strikes_absorbed": _f("fighter_sig_strikes_absorbed"),
        "td_landed":      _f("fighter_td_landed"),
        "td_acc":         _f("fighter_td_acc"),
        "sig_strike_acc": _f("fighter_sig_strike_acc"),
        "dec_rate_overall": _f("fighter_dec_rate_overall", 0.5),
    }


def fuzzy_lookup(name):
    """Try multiple name formats to find fighter in dataset."""
    # 1. Exact
    rec = fighter_record(name)
    if rec:
        return name, rec

    if _fights_df is None:
        return None, None

    name_lower = name.lower()
    all_fighters = _fights_df["fighter"].unique()

    # 2. Substring
    matches = [f for f in all_fighters if name_lower in f.lower()]
    if len(matches) == 1:
        return matches[0], fighter_record(matches[0])

    # 3. Last name match
    last = name_lower.split()[-1]
    matches2 = [f for f in all_fighters if last in f.lower().split()]
    if len(matches2) == 1:
        return matches2[0], fighter_record(matches2[0])

    return None, None


def stats_from_db(name):
    """Get physical stats from fighter_stats.csv (more reliable for height/reach/stance)."""
    if _stats_df is None:
        return {}
    if name in _stats_df.index:
        row = _stats_df.loc[name]
        result = {}
        if row["altura_cm"] is not None and not (isinstance(row["altura_cm"], float) and np.isnan(row["altura_cm"])):
            result["height"] = round(float(row["altura_cm"]), 1)
        if row["envergadura_cm"] is not None and not (isinstance(row["envergadura_cm"], float) and np.isnan(row["envergadura_cm"])):
            result["reach"] = round(float(row["envergadura_cm"]), 1)
        result["stance"] = row["stance_norm"]
        return result

    # fuzzy in stats
    name_lower = name.lower()
    matches = [n for n in _stats_df.index if name_lower in n.lower()]
    if len(matches) == 1:
        return stats_from_db(matches[0])
    return {}


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------

def _taxa(num, den):
    return num / den if den > 0 else 0.0


def build_feature_vector(f, o):
    f_taxa_ko  = _taxa(f["ko"],  f["wins"])
    f_taxa_sub = _taxa(f["sub"], f["wins"])
    f_taxa_dec = _taxa(max(f["wins"] - f["ko"] - f["sub"], 0), f["wins"])
    o_taxa_ko  = _taxa(o["ko"],  o["wins"])
    o_taxa_sub = _taxa(o["sub"], o["wins"])
    o_taxa_dec = _taxa(max(o["wins"] - o["ko"] - o["sub"], 0), o["wins"])

    row = {
        # Group 1 — basic differentials
        "d_idade":        f["age"]    - o["age"],
        "d_altura":       f["height"] - o["height"],
        "d_envergadura":  f["reach"]  - o["reach"],
        "d_vitorias":     f["wins"]   - o["wins"],
        "d_derrotas":     f["losses"] - o["losses"],
        "d_lutas_totais": (f["wins"] + f["losses"]) - (o["wins"] + o["losses"]),
        "d_taxa_ko":      f_taxa_ko  - o_taxa_ko,
        "d_taxa_sub":     f_taxa_sub - o_taxa_sub,
        "d_taxa_dec":     f_taxa_dec - o_taxa_dec,
        "d_ko_losses":         f.get("ko_losses", 0)         - o.get("ko_losses", 0),
        "d_dec_rate_overall":  f.get("dec_rate_overall", 0.5) - o.get("dec_rate_overall", 0.5),
        # Group 2 — recent form
        "d_L5Y_winrate":  f.get("L5Y_winrate", 0.5) - o.get("L5Y_winrate", 0.5),
        "d_L2Y_winrate":  f.get("L2Y_winrate", 0.5) - o.get("L2Y_winrate", 0.5),
        # Group 3 — fight stats
        "d_sig_strikes_landed":   f.get("sig_strikes_landed", 0)   - o.get("sig_strikes_landed", 0),
        "d_sig_strikes_absorbed": f.get("sig_strikes_absorbed", 0) - o.get("sig_strikes_absorbed", 0),
        "d_td_landed":            f.get("td_landed", 0)            - o.get("td_landed", 0),
        "d_td_acc":               f.get("td_acc", 0)               - o.get("td_acc", 0),
        "d_sig_strike_acc":       f.get("sig_strike_acc", 0)       - o.get("sig_strike_acc", 0),
    }
    for e in config.ESTILOS:
        row[f"d_stance_{e}"] = (1 if f["stance"] == e else 0) - (1 if o["stance"] == e else 0)
    row["estilos_diferentes"] = int(f["stance"] != o["stance"])

    return np.array([row[feat] for feat in config.FEATURES], dtype=float)


def predict(f_stats, o_stats, title_fight: bool = False):
    X = build_feature_vector(f_stats, o_stats).reshape(1, -1)

    if title_fight:
        # Boost the d_dec_rate_overall feature (endurance proxy) for 5-round fights.
        # The boost amplifies the advantage of the fighter with more decision experience.
        idx = config.FEATURES.index("d_dec_rate_overall")
        X[0, idx] *= config.TITLE_FIGHT_DEC_BOOST

    probs = {}
    for key, model in _models.items():
        probs[key] = float(model.predict_proba(X)[0, 1])
    ensemble = float(np.mean(list(probs.values())))
    return {"models": probs, "ensemble": ensemble}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/api/rankings")
def api_rankings():
    try:
        data = get_rankings()
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/fighter/<path:name>")
def api_fighter(name):
    """Lookup fighter stats by display name (from UFC website)."""
    matched_name, rec = fuzzy_lookup(name)
    if rec is None:
        return jsonify({"ok": False, "error": f"Fighter '{name}' not found"}), 404

    # Merge with fighter_stats.csv for cleaner physical data
    phys = stats_from_db(matched_name)
    rec.update(phys)
    rec["matched_name"] = matched_name
    return jsonify({"ok": True, "data": rec})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True)
    try:
        red = body["red"]
        blue = body["blue"]

        # Defaults for missing fields
        for side in (red, blue):
            side.setdefault("age", 30.0)
            side.setdefault("height", 177.8)
            side.setdefault("reach", 182.9)
            side.setdefault("stance", "orthodox")
            side.setdefault("wins", 0)
            side.setdefault("losses", 0)
            side.setdefault("ko", 0)
            side.setdefault("sub", 0)
            side.setdefault("ko_losses", 0)
            side.setdefault("L5Y_winrate", 0.5)
            side.setdefault("L2Y_winrate", 0.5)
            side.setdefault("sig_strikes_landed", 0.0)
            side.setdefault("sig_strikes_absorbed", 0.0)
            side.setdefault("td_landed", 0.0)
            side.setdefault("td_acc", 0.0)
            side.setdefault("sig_strike_acc", 0.0)
            side.setdefault("dec_rate_overall", 0.5)

        title_fight = bool(body.get("title_fight", False))
        result = predict(red, blue, title_fight=title_fight)
        return jsonify({"ok": True, "data": result})
    except KeyError as e:
        return jsonify({"ok": False, "error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("MuchaLutcha server starting on http://localhost:5000")
    app.run(debug=False, port=5000)
