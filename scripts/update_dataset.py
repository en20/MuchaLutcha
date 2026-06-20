"""
Scrape UFC events from Dec 15, 2024 onward and append to data/raw/ufc_fights.csv.

Strategy:
- Get all events from ufcstats.com after the dataset cutoff
- For each event, scrape fights (winner, loser, division, method, round, time)
- For each fighter, scrape their career stats from ufcstats fighter page
- Since we can't reconstruct pre-fight state for avg stats, we use current career averages
  (reasonable approximation — career avgs change little fight-to-fight for established fighters)
- Write rows matching the 143-column schema of ufc_fights.csv
"""

import asyncio
import csv
import json
import re
import sys
from datetime import datetime, date
from pathlib import Path

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
CUTOFF = date(2024, 12, 14)
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ufc_fights.csv"
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "fighter_cache.json"

# All 143 column names in order
COLUMNS = [
    "date","division","fighter","opponent","result","method",
    "fighter_wins","fighter_losses","fighter_age","fighter_height","fighter_reach",
    "fighter_L5Y_wins","fighter_L5Y_losses","fighter_L2Y_wins","fighter_L2Y_losses",
    "fighter_ko_wins","fighter_ko_losses",
    "fighter_L5Y_ko_wins","fighter_L5Y_ko_losses","fighter_L2Y_ko_wins","fighter_L2Y_ko_losses",
    "fighter_sub_wins","fighter_sub_losses",
    "fighter_L5Y_sub_wins","fighter_L5Y_sub_losses","fighter_L2Y_sub_wins","fighter_L2Y_sub_losses",
    "fighter_inf_knockdowns_avg","fighter_inf_pass_avg","fighter_inf_reversals_avg",
    "fighter_inf_sub_attempts_avg","fighter_inf_takedowns_landed_avg","fighter_inf_takedowns_attempts_avg",
    "fighter_inf_sig_strikes_landed_avg","fighter_inf_sig_strikes_attempts_avg",
    "fighter_inf_total_strikes_landed_avg","fighter_inf_total_strikes_attempts_avg",
    "fighter_inf_head_strikes_landed_avg","fighter_inf_head_strikes_attempts_avg",
    "fighter_inf_body_strikes_landed_avg","fighter_inf_body_strikes_attempts_avg",
    "fighter_inf_leg_strikes_landed_avg","fighter_inf_leg_strikes_attempts_avg",
    "fighter_inf_distance_strikes_landed_avg","fighter_inf_distance_strikes_attempts_avg",
    "fighter_inf_clinch_strikes_landed_avg","fighter_inf_clinch_strikes_attempts_avg",
    "fighter_inf_ground_strikes_landed_avg","fighter_inf_ground_strikes_attempts_avg",
    "fighter_abs_knockdowns_avg","fighter_abs_pass_avg","fighter_abs_reversals_avg",
    "fighter_abs_sub_attempts_avg","fighter_abs_takedowns_landed_avg","fighter_abs_takedowns_attempts_avg",
    "fighter_abs_sig_strikes_landed_avg","fighter_abs_sig_strikes_attempts_avg",
    "fighter_abs_total_strikes_landed_avg","fighter_abs_total_strikes_attempts_avg",
    "fighter_abs_head_strikes_landed_avg","fighter_abs_head_strikes_attempts_avg",
    "fighter_abs_body_strikes_landed_avg","fighter_abs_body_strikes_attempts_avg",
    "fighter_abs_leg_strikes_landed_avg","fighter_abs_leg_strikes_attempts_avg",
    "fighter_abs_distance_strikes_landed_avg","fighter_abs_distance_strikes_attempts_avg",
    "fighter_abs_clinch_strikes_landed_avg","fighter_abs_clinch_strikes_attempts_avg",
    "fighter_abs_ground_strikes_landed_avg","fighter_abs_ground_strikes_attempts_avg",
    "opponent_wins","opponent_losses","opponent_age","opponent_height","opponent_reach",
    "opponent_L5Y_wins","opponent_L5Y_losses","opponent_L2Y_wins","opponent_L2Y_losses",
    "opponent_ko_wins","opponent_ko_losses",
    "opponent_L5Y_ko_wins","opponent_L5Y_ko_losses","opponent_L2Y_ko_wins","opponent_L2Y_ko_losses",
    "opponent_sub_wins","opponent_sub_losses",
    "opponent_L5Y_sub_wins","opponent_L5Y_sub_losses","opponent_L2Y_sub_wins","opponent_L2Y_sub_losses",
    "opponent_inf_knockdowns_avg","opponent_inf_pass_avg","opponent_inf_reversals_avg",
    "opponent_inf_sub_attempts_avg","opponent_inf_takedowns_landed_avg","opponent_inf_takedowns_attempts_avg",
    "opponent_inf_sig_strikes_landed_avg","opponent_inf_sig_strikes_attempts_avg",
    "opponent_inf_total_strikes_landed_avg","opponent_inf_total_strikes_attempts_avg",
    "opponent_inf_head_strikes_landed_avg","opponent_inf_head_strikes_attempts_avg",
    "opponent_inf_body_strikes_landed_avg","opponent_inf_body_strikes_attempts_avg",
    "opponent_inf_leg_strikes_landed_avg","opponent_inf_leg_strikes_attempts_avg",
    "opponent_inf_distance_strikes_landed_avg","opponent_inf_distance_strikes_attempts_avg",
    "opponent_inf_clinch_strikes_landed_avg","opponent_inf_clinch_strikes_attempts_avg",
    "opponent_inf_ground_strikes_landed_avg","opponent_inf_ground_strikes_attempts_avg",
    "opponent_abs_knockdowns_avg","opponent_abs_pass_avg","opponent_abs_reversals_avg",
    "opponent_abs_sub_attempts_avg","opponent_abs_takedowns_landed_avg","opponent_abs_takedowns_attempts_avg",
    "opponent_abs_sig_strikes_landed_avg","opponent_abs_sig_strikes_attempts_avg",
    "opponent_abs_total_strikes_landed_avg","opponent_abs_total_strikes_attempts_avg",
    "opponent_abs_head_strikes_landed_avg","opponent_abs_head_strikes_attempts_avg",
    "opponent_abs_body_strikes_landed_avg","opponent_abs_body_strikes_attempts_avg",
    "opponent_abs_leg_strikes_landed_avg","opponent_abs_leg_strikes_attempts_avg",
    "opponent_abs_distance_strikes_landed_avg","opponent_abs_distance_strikes_attempts_avg",
    "opponent_abs_clinch_strikes_landed_avg","opponent_abs_clinch_strikes_attempts_avg",
    "opponent_abs_ground_strikes_landed_avg","opponent_abs_ground_strikes_attempts_avg",
    "fighter_stance","opponent_stance",
    "1-fight_math","6-fight_math","4-fighter_score_diff","9-fighter_score_diff","15-fighter_score_diff",
]

assert len(COLUMNS) == 143, f"Expected 143 columns, got {len(COLUMNS)}"

# ---------------------------------------------------------------------------

def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}

def save_cache(cache: dict):
    CACHE_PATH.write_text(json.dumps(cache, indent=2))

def parse_height_cm(s: str) -> str:
    """'6\' 2"' → cm as float string, or 'unknown'"""
    if not s:
        return "unknown"
    m = re.match(r"(\d+)'\s*(\d+)", s)
    if m:
        return str((int(m.group(1)) * 12 + int(m.group(2))) * 2.54)
    return "unknown"

def parse_reach_cm(s: str) -> str:
    """'75"' → cm as float string, or 'unknown'"""
    if not s:
        return "unknown"
    m = re.match(r"([\d.]+)", s)
    if m:
        return str(float(m.group(1)) * 2.54)
    return "unknown"

def parse_pct(s: str) -> str:
    """'55%' → '0.55'"""
    m = re.match(r"([\d.]+)%", s.strip())
    return str(float(m.group(1)) / 100) if m else "0.0"

def safe_float(s: str) -> str:
    s = s.strip()
    try:
        return str(float(s))
    except Exception:
        return "0.0"

def normalize_method(method: str) -> str:
    """Normalize fight-end method to match existing dataset style."""
    m = method.strip().upper()
    if "KO" in m or "TKO" in m:
        return "KO/TKO"
    if "SUB" in m:
        return "SUB"
    if "DEC" in m or "DECISION" in m:
        return "DEC"
    if "NC" in m or "NO CONTEST" in m:
        return "NC"
    return method.strip()

def normalize_stance(s: str) -> str:
    s = s.strip().lower()
    return s if s in ("orthodox", "southpaw", "switch") else "other"

# ---------------------------------------------------------------------------
# ufcstats scraping helpers
# ---------------------------------------------------------------------------

async def get_page_html(page, url: str, wait: str = "networkidle") -> str:
    await page.goto(url, timeout=45000)
    await page.wait_for_load_state(wait)
    return await page.content()

async def scrape_events_after_cutoff(page) -> list[dict]:
    """Return list of {name, url, date} for events after CUTOFF."""
    html = await get_page_html(page, "http://ufcstats.com/statistics/events/completed?page=all")
    soup = BeautifulSoup(html, "html.parser")

    events = []
    for row in soup.select("tr.b-statistics__table-row"):
        link = row.select_one("a.b-link")
        date_td = row.select_one("span.b-statistics__date")
        if not link or not date_td:
            continue
        date_str = date_td.get_text(strip=True)
        try:
            event_date = datetime.strptime(date_str, "%B %d, %Y").date()
        except Exception:
            continue
        if event_date <= CUTOFF:
            continue
        events.append({
            "name": link.get_text(strip=True),
            "url": link["href"],
            "date": event_date,
            "date_str": event_date.strftime("%B %d, %Y"),
        })

    events.sort(key=lambda e: e["date"])
    print(f"Found {len(events)} events after {CUTOFF}")
    return events

async def scrape_event_fights(page, event: dict) -> list[dict]:
    """Scrape all fights from a single event page."""
    html = await get_page_html(page, event["url"])
    soup = BeautifulSoup(html, "html.parser")

    fights = []
    table = soup.select_one("table.b-fight-details__table")
    if not table:
        print(f"  ⚠ No fight table in {event['name']}")
        return fights

    rows = table.select("tr.b-fight-details__table-row")[1:]  # skip header
    for row in rows:
        cols = row.select("td.b-fight-details__table-col")
        if len(cols) < 8:
            continue

        # Col 0: result + fighter names
        fighters_td = cols[1]
        fighter_links = fighters_td.select("a.b-link")
        if len(fighter_links) < 2:
            continue
        f1_name = fighter_links[0].get_text(strip=True)
        f2_name = fighter_links[1].get_text(strip=True)
        f1_url  = fighter_links[0]["href"]
        f2_url  = fighter_links[1]["href"]

        # Col 0: result flag — b-flag_style_green = win, b-flag_style_red = loss, others = NC
        result_col = cols[0]
        flag = result_col.select_one("a.b-flag")
        flag_class = flag.get("class", []) if flag else []
        flag_text  = " ".join(flag_class)
        if "green" in flag_text:
            f1_result, f2_result = "W", "L"
        elif "red" in flag_text:
            f1_result, f2_result = "L", "W"
        else:
            f1_result = f2_result = "NC"

        # Col 7: division
        division = cols[6].get_text(strip=True) if len(cols) > 6 else "unknown"

        # Col 7: method
        method_raw = cols[7].get_text(strip=True) if len(cols) > 7 else ""
        method = normalize_method(method_raw)

        # Col 8: round
        rnd = cols[8].get_text(strip=True) if len(cols) > 8 else "0"

        # Col 9: time
        time_ = cols[9].get_text(strip=True) if len(cols) > 9 else "0:00"

        # Skip NC/draw
        if f1_result == "NC":
            continue

        fights.append({
            "date_str": event["date_str"],
            "division": division,
            "f1_name": f1_name, "f1_url": f1_url, "f1_result": f1_result,
            "f2_name": f2_name, "f2_url": f2_url, "f2_result": f2_result,
            "method": method, "round": rnd, "time": time_,
        })

    print(f"  {event['name']}: {len(fights)} valid fights")
    return fights

# ---------------------------------------------------------------------------
# Fighter stats scraping
# ---------------------------------------------------------------------------

def _extract_stat_value(soup: BeautifulSoup, label: str) -> str:
    """Find a stat by its label text on the fighter page."""
    for li in soup.select("li.b-list__box-list-item"):
        text = li.get_text(" ", strip=True)
        if label.lower() in text.lower():
            parts = text.split(":")
            if len(parts) >= 2:
                return parts[-1].strip()
    return ""

async def scrape_fighter_stats(page, name: str, url: str, cache: dict) -> dict:
    """Return fighter stats dict. Uses cache to avoid redundant requests."""
    if url in cache:
        return cache[url]

    try:
        html = await get_page_html(page, url)
    except Exception as e:
        print(f"    ⚠ Failed to load {url}: {e}")
        cache[url] = {}
        return {}

    soup = BeautifulSoup(html, "html.parser")
    # Flat text tokens split on whitespace/pipes for robust label→value lookup
    tokens = [t for t in re.split(r"[\|\n]+", soup.body.get_text(separator="|")) if t.strip()]

    def next_token(label: str) -> str:
        """Return the token immediately after the one matching label (case-insensitive)."""
        for i, t in enumerate(tokens):
            if label.lower() in t.strip().lower() and i + 1 < len(tokens):
                return tokens[i + 1].strip()
        return ""

    # Record: look for "Record: W-L-D" token
    wins = losses = draws = 0
    for t in tokens:
        m = re.search(r"Record:\s*(\d+)-(\d+)-(\d+)", t)
        if m:
            wins, losses, draws = int(m.group(1)), int(m.group(2)), int(m.group(3))
            break

    height_raw = next_token("Height:")
    reach_raw  = next_token("Reach:")
    dob_raw    = next_token("DOB:")
    stance_raw = next_token("STANCE:")

    # Age from DOB
    age = "unknown"
    if dob_raw and dob_raw not in ("--", ""):
        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                dob = datetime.strptime(dob_raw, fmt)
                age = str(int((datetime.today() - dob).days / 365.25))
                break
            except Exception:
                pass

    # Career stats — appear in a predictable order after "Career statistics:"
    slpm    = safe_float(next_token("SLpM:"))
    str_acc = parse_pct(next_token("Str. Acc.:"))
    sapm    = safe_float(next_token("SApM:"))
    td_avg  = safe_float(next_token("TD Avg.:"))
    td_acc  = parse_pct(next_token("TD Acc.:"))
    sub_avg = safe_float(next_token("Sub. Avg.:"))

    # Career win breakdown from fight history table
    ko_wins = ko_losses = sub_wins = sub_losses = 0
    for row in soup.select("tr.b-fight-details__table-row")[1:]:
        cols = row.select("td")
        if len(cols) < 8:
            continue
        result = cols[0].get_text(strip=True).lower()
        method_text = cols[7].get_text(strip=True).upper() if len(cols) > 7 else ""
        is_ko  = "KO" in method_text or "TKO" in method_text
        is_sub = "SUB" in method_text
        if result == "win":
            if is_ko:  ko_wins  += 1
            if is_sub: sub_wins += 1
        elif result == "loss":
            if is_ko:  ko_losses  += 1
            if is_sub: sub_losses += 1

    # Convert per-minute stats to per-fight averages (use 15 min as career avg fight time)
    FIGHT_MIN = 15.0
    sig_l_avg = str(float(slpm) * FIGHT_MIN)
    str_acc_f = float(str_acc) if float(str_acc) > 0 else 0.55
    sig_a_avg = str(float(slpm) * FIGHT_MIN / str_acc_f)
    abs_l_avg = str(float(sapm) * FIGHT_MIN)
    td_l_avg  = str(float(td_avg))
    td_acc_f  = float(td_acc) if float(td_acc) > 0 else 0.5
    td_a_avg  = str(float(td_avg) / td_acc_f)

    stats = {
        "wins": str(wins), "losses": str(losses),
        "age": age,
        "height": parse_height_cm(height_raw),
        "reach": parse_reach_cm(reach_raw),
        "stance": normalize_stance(stance_raw),
        "ko_wins": str(ko_wins), "ko_losses": str(ko_losses),
        "sub_wins": str(sub_wins), "sub_losses": str(sub_losses),
        "inf_sig_strikes_landed_avg": sig_l_avg,
        "inf_sig_strikes_attempts_avg": sig_a_avg,
        "abs_sig_strikes_landed_avg": abs_l_avg,
        "inf_takedowns_landed_avg": td_l_avg,
        "inf_takedowns_attempts_avg": td_a_avg,
        "inf_sub_attempts_avg": str(float(sub_avg)),
        "td_acc": td_acc,
    }

    cache[url] = stats
    return stats

def build_fighter_row_values(stats: dict, prefix: str, fight_date: str,
                              year_cutoff5: int, year_cutoff2: int) -> dict:
    """Build the ~70 columns for one fighter side (fighter_ or opponent_)."""
    wins   = int(float(stats.get("wins", 0) or 0))
    losses = int(float(stats.get("losses", 0) or 0))
    ko_w   = int(float(stats.get("ko_wins", 0) or 0))
    ko_l   = int(float(stats.get("ko_losses", 0) or 0))
    sub_w  = int(float(stats.get("sub_wins", 0) or 0))
    sub_l  = int(float(stats.get("sub_losses", 0) or 0))

    # L5Y/L2Y: we don't have historical data, use total record as approximation
    # (same as what a fighter would have if all fights are recent)
    # This is a limitation but acceptable for recent fighters
    l5y_w = wins; l5y_l = losses
    l2y_w = wins; l2y_l = losses
    l5y_ko_w = ko_w; l5y_ko_l = ko_l; l2y_ko_w = ko_w; l2y_ko_l = ko_l
    l5y_sub_w = sub_w; l5y_sub_l = sub_l; l2y_sub_w = sub_w; l2y_sub_l = sub_l

    sig_l  = stats.get("inf_sig_strikes_landed_avg", "0.0")
    sig_a  = stats.get("inf_sig_strikes_attempts_avg", "0.0")
    abs_l  = stats.get("abs_sig_strikes_landed_avg", "0.0")
    td_l   = stats.get("inf_takedowns_landed_avg", "0.0")
    td_a   = stats.get("inf_takedowns_attempts_avg", "0.0")
    sub_a  = stats.get("inf_sub_attempts_avg", "0.0")

    row = {
        f"{prefix}_wins":    str(wins),
        f"{prefix}_losses":  str(losses),
        f"{prefix}_age":     stats.get("age", "unknown"),
        f"{prefix}_height":  stats.get("height", "unknown"),
        f"{prefix}_reach":   stats.get("reach", "unknown"),
        f"{prefix}_L5Y_wins":   str(l5y_w), f"{prefix}_L5Y_losses": str(l5y_l),
        f"{prefix}_L2Y_wins":   str(l2y_w), f"{prefix}_L2Y_losses": str(l2y_l),
        f"{prefix}_ko_wins":    str(ko_w),  f"{prefix}_ko_losses":  str(ko_l),
        f"{prefix}_L5Y_ko_wins": str(l5y_ko_w), f"{prefix}_L5Y_ko_losses": str(l5y_ko_l),
        f"{prefix}_L2Y_ko_wins": str(l2y_ko_w), f"{prefix}_L2Y_ko_losses": str(l2y_ko_l),
        f"{prefix}_sub_wins":    str(sub_w), f"{prefix}_sub_losses":  str(sub_l),
        f"{prefix}_L5Y_sub_wins": str(l5y_sub_w), f"{prefix}_L5Y_sub_losses": str(l5y_sub_l),
        f"{prefix}_L2Y_sub_wins": str(l2y_sub_w), f"{prefix}_L2Y_sub_losses": str(l2y_sub_l),
        # Inflicted averages (per fight)
        f"{prefix}_inf_knockdowns_avg": "0.0",
        f"{prefix}_inf_pass_avg": "0.0",
        f"{prefix}_inf_reversals_avg": "0.0",
        f"{prefix}_inf_sub_attempts_avg": sub_a,
        f"{prefix}_inf_takedowns_landed_avg":   td_l,
        f"{prefix}_inf_takedowns_attempts_avg":  td_a,
        f"{prefix}_inf_sig_strikes_landed_avg":  sig_l,
        f"{prefix}_inf_sig_strikes_attempts_avg": sig_a,
        f"{prefix}_inf_total_strikes_landed_avg": sig_l,
        f"{prefix}_inf_total_strikes_attempts_avg": sig_a,
        f"{prefix}_inf_head_strikes_landed_avg": "0.0",
        f"{prefix}_inf_head_strikes_attempts_avg": "0.0",
        f"{prefix}_inf_body_strikes_landed_avg": "0.0",
        f"{prefix}_inf_body_strikes_attempts_avg": "0.0",
        f"{prefix}_inf_leg_strikes_landed_avg": "0.0",
        f"{prefix}_inf_leg_strikes_attempts_avg": "0.0",
        f"{prefix}_inf_distance_strikes_landed_avg": "0.0",
        f"{prefix}_inf_distance_strikes_attempts_avg": "0.0",
        f"{prefix}_inf_clinch_strikes_landed_avg": "0.0",
        f"{prefix}_inf_clinch_strikes_attempts_avg": "0.0",
        f"{prefix}_inf_ground_strikes_landed_avg": "0.0",
        f"{prefix}_inf_ground_strikes_attempts_avg": "0.0",
        # Absorbed averages
        f"{prefix}_abs_knockdowns_avg": "0.0",
        f"{prefix}_abs_pass_avg": "0.0",
        f"{prefix}_abs_reversals_avg": "0.0",
        f"{prefix}_abs_sub_attempts_avg": "0.0",
        f"{prefix}_abs_takedowns_landed_avg": "0.0",
        f"{prefix}_abs_takedowns_attempts_avg": "0.0",
        f"{prefix}_abs_sig_strikes_landed_avg": abs_l,
        f"{prefix}_abs_sig_strikes_attempts_avg": abs_l,
        f"{prefix}_abs_total_strikes_landed_avg": abs_l,
        f"{prefix}_abs_total_strikes_attempts_avg": abs_l,
        f"{prefix}_abs_head_strikes_landed_avg": "0.0",
        f"{prefix}_abs_head_strikes_attempts_avg": "0.0",
        f"{prefix}_abs_body_strikes_landed_avg": "0.0",
        f"{prefix}_abs_body_strikes_attempts_avg": "0.0",
        f"{prefix}_abs_leg_strikes_landed_avg": "0.0",
        f"{prefix}_abs_leg_strikes_attempts_avg": "0.0",
        f"{prefix}_abs_distance_strikes_landed_avg": "0.0",
        f"{prefix}_abs_distance_strikes_attempts_avg": "0.0",
        f"{prefix}_abs_clinch_strikes_landed_avg": "0.0",
        f"{prefix}_abs_clinch_strikes_attempts_avg": "0.0",
        f"{prefix}_abs_ground_strikes_landed_avg": "0.0",
        f"{prefix}_abs_ground_strikes_attempts_avg": "0.0",
        f"{prefix}_stance": stats.get("stance", "other"),
    }
    return row

def fight_to_csv_rows(fight: dict, f1_stats: dict, f2_stats: dict) -> list[dict]:
    """Convert a fight dict + stats into 2 CSV rows (fighter perspective + opponent)."""
    rows = []
    pairs = [
        (fight["f1_name"], fight["f2_name"], fight["f1_result"], f1_stats, f2_stats),
        (fight["f2_name"], fight["f1_name"], fight["f2_result"], f2_stats, f1_stats),
    ]
    for fighter, opponent, result, f_stats, o_stats in pairs:
        row = {
            "date": fight["date_str"],
            "division": fight["division"],
            "fighter": fighter,
            "opponent": opponent,
            "result": result,
            "method": fight["method"],
        }
        row.update(build_fighter_row_values(f_stats, "fighter", fight["date_str"], 0, 0))
        row.update(build_fighter_row_values(o_stats, "opponent", fight["date_str"], 0, 0))
        # Trailing columns (fight math scores — not used in our pipeline, fill 0)
        row["1-fight_math"]         = "0"
        row["6-fight_math"]         = "0"
        row["4-fighter_score_diff"] = "0"
        row["9-fighter_score_diff"] = "0"
        row["15-fighter_score_diff"]= "0"
        rows.append(row)
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    cache = load_cache()
    print(f"Loaded fighter cache: {len(cache)} entries")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # 1) Get all events after cutoff
        events = await scrape_events_after_cutoff(page)
        if not events:
            print("No new events found.")
            await browser.close()
            return

        # 2) Scrape each event
        all_new_rows = []
        for i, event in enumerate(events):
            print(f"\n[{i+1}/{len(events)}] {event['name']} ({event['date_str']})")
            fights = await scrape_event_fights(page, event)

            for fight in fights:
                # Fetch fighter stats
                f1_stats = await scrape_fighter_stats(page, fight["f1_name"], fight["f1_url"], cache)
                f2_stats = await scrape_fighter_stats(page, fight["f2_name"], fight["f2_url"], cache)

                rows = fight_to_csv_rows(fight, f1_stats, f2_stats)
                all_new_rows.extend(rows)

            # Save cache periodically
            if (i + 1) % 5 == 0:
                save_cache(cache)
                print(f"  [cache saved, {len(all_new_rows)} rows so far]")

        await browser.close()

    save_cache(cache)
    print(f"\nTotal new rows to append: {len(all_new_rows)}")

    # 3) Append to CSV
    if not all_new_rows:
        print("Nothing to append.")
        return

    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for row in all_new_rows:
            # Ensure all columns present
            for col in COLUMNS:
                if col not in row:
                    row[col] = "0.0"
            writer.writerow({col: row[col] for col in COLUMNS})

    print(f"Appended {len(all_new_rows)} rows to {CSV_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
