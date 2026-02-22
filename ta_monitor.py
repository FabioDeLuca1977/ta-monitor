#!/usr/bin/env python3
"""
TA Monitor — Monitoraggio Posizioni HR / Talent Acquisition Roma
Maieutike srl — v2.1 (GitHub Actions edition)

Uso:
    python ta_monitor.py                              # Default: ultime 72h
    python ta_monitor.py --hours-old 168              # Ultima settimana
    python ta_monitor.py --search-terms "HR manager"  # Keywords custom
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from rapidfuzz import fuzz

# === PATHS ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# FILTRO DI RILEVANZA
# Solo posizioni HR / Talent Acquisition / Recruiting /
# Gestione filiali agenzie somministrazione
# ============================================================

RELEVANT_TITLE_PATTERNS = [
    # Talent Acquisition
    r"talent.?acquisition",
    # HR Business Partner
    r"hr\s*b\.?p\.?",
    r"human\s*resource.*business\s*partner",
    r"hr\s+business\s+partner",
    # HR Generalist
    r"hr\s+generalist",
    r"human\s*resource.*generalist",
    # Recruiting / Recruiter / Selezionatrice/ore
    r"recruit(?:er|ing|ment)",
    r"selezionat(?:ore|rice|rici)",
    r"selezione\s+(?:del\s+)?personale",
    r"responsabile\s+selezione",
    r"addett[oa]\s+(?:alla?\s+)?selezione",
    r"specialista\s+(?:della?\s+)?selezione",
    # HR Specialist in ambito selezione
    r"hr\s+specialist.*(?:selezione|recruiting|ricerca)",
    r"specialist[a]?\s+(?:hr|risorse\s+umane)",
    # Head hunter
    r"head\s*hunt(?:er|ing)",
    # Gestione filiali agenzie per il lavoro / somministrazione
    r"respons.*filiale",
    r"branch\s*manager",
    r"dirett.*filiale",
    r"gestione\s+filiale",
    r"responsabile\s+(?:di\s+)?filiale",
    r"area\s+manager.*(?:somministr|staffing|apl|agenzia)",
    # Agenzie per il lavoro — ruoli operativi
    r"(?:account|consultant).*(?:apl|somministr|staffing|agenzia.*lavoro)",
    r"hr\s+consultant",
]

_RELEVANT_COMPILED = [re.compile(p, re.IGNORECASE) for p in RELEVANT_TITLE_PATTERNS]

BROAD_KEYWORDS = [
    "talent", "recruiter", "recruiting", "recruitment",
    "selezione", "selezionatrice", "selezionatore",
    "hrbp", "hr generalist", "hr specialist",
    "head hunter", "headhunter",
    "filiale", "branch manager",
    "risorse umane", "human resources",
    "hr consultant",
]

# Blacklist: titoli NON pertinenti
TITLE_BLACKLIST = [
    r"sviluppat", r"developer", r"software\s+engineer",
    r"full\s*stack", r"front\s*end", r"back\s*end",
    r"data\s+(?:engineer|scientist|analyst)",
    r"devops", r"sys\s*admin", r"cloud.*(?:architect|engineer)",
    r"manutenz", r"autogrill", r"camerier", r"barista",
    r"cuoc[oa]", r"magazzin", r"operai[oa]", r"muratore",
    r"idraulic", r"elettricist", r"infermier", r"farmacist",
    r"contabil", r"segretari[oa]", r"receptionist",
    r"commess[oa]", r"cassier", r"addet.*vendita",
    r"graphic\s*design", r"marketing\s+(?:specialist|manager)",
    r"social\s+media", r"seo\s", r"copywriter",
    r"consulen.*(?:finanzi|fiscal|legale|immobili)",
    r"agente.*(?:commerc|immobili|assicur)",
]

_BLACKLIST_COMPILED = [re.compile(p, re.IGNORECASE) for p in TITLE_BLACKLIST]


def is_relevant_job(title, description=""):
    if not title:
        return False
    title_str = str(title).strip()

    # Blacklist
    for bp in _BLACKLIST_COMPILED:
        if bp.search(title_str):
            return False

    # Pattern specifici
    for rp in _RELEVANT_COMPILED:
        if rp.search(title_str):
            return True

    # Keyword broad
    title_lower = title_str.lower()
    for kw in BROAD_KEYWORDS:
        if kw.lower() in title_lower:
            return True

    # Fallback: analisi descrizione
    if description:
        desc_lower = str(description).lower()
        hr_signals = [
            "talent acquisition", "recruiting", "selezione del personale",
            "ricerca e selezione", "hr business partner", "somministrazione",
            "agenzia per il lavoro", "staffing",
        ]
        title_has_hr = any(w in title_lower for w in ["hr", "human", "risorse", "personale", "staff"])
        desc_has_signal = sum(1 for s in hr_signals if s in desc_lower)
        if title_has_hr and desc_has_signal >= 2:
            return True

    return False


# === LOGGING ===

def setup_logging():
    logger = logging.getLogger("ta_monitor")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(LOG_DIR / "ta_monitor.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

log = setup_logging()

def load_config():
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# === DATABASE ===

def init_db():
    db_path = DATA_DIR / "ta_monitor.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            company TEXT,
            location TEXT,
            channel TEXT,
            url TEXT,
            description TEXT,
            salary_min REAL,
            salary_max REAL,
            job_type TEXT,
            date_posted TEXT,
            date_scraped TEXT NOT NULL,
            search_term TEXT,
            is_new INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            total_found INTEGER,
            new_found INTEGER,
            duplicates_removed INTEGER,
            filtered_irrelevant INTEGER,
            status TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_date ON jobs(date_scraped);
        CREATE INDEX IF NOT EXISTS idx_channel ON jobs(channel);
    """)
    conn.commit()
    return conn

def job_id(title, company, url):
    raw = f"{(title or '').lower().strip()}|{(company or '').lower().strip()}|{(url or '').strip()}"
    return hashlib.md5(raw.encode()).hexdigest()

# === JOBSPY ===

def scrape_jobspy(config, hours_old):
    try:
        from jobspy import scrape_jobs
    except ImportError:
        log.error("python-jobspy non installato!")
        return pd.DataFrame()

    sites = config.get("jobspy_sites", ["indeed", "linkedin", "google"])
    location = config.get("location", "Roma, Lazio, Italia")
    proxies = config.get("proxies", []) or []
    results = []

    for term in config.get("search_terms", []):
        log.info(f"  JobSpy [{','.join(sites)}] → '{term}'")
        try:
            df = scrape_jobs(
                site_name=sites,
                search_term=term,
                google_search_term=f"{term} lavoro Roma Italia",
                location=location,
                results_wanted=config.get("jobspy_params", {}).get("results_wanted", 50),
                hours_old=hours_old,
                country_indeed="Italy",
                linkedin_fetch_description=config.get("jobspy_params", {}).get("linkedin_fetch_description", False),
                proxies=proxies if proxies else None,
            )
            if not df.empty:
                df["search_term"] = term
                results.append(df)
                log.info(f"    ✓ {len(df)} risultati")
            else:
                log.info(f"    · 0 risultati")
        except Exception as e:
            log.warning(f"    ✗ Errore: {e}")

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

# === CUSTOM SCRAPERS ===

def scrape_custom(config):
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning("requests/bs4 mancanti, skip scraper custom")
        return []

    custom_cfg = config.get("custom_scrapers", {})
    jobs = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0",
        "Accept-Language": "it-IT,it;q=0.9",
    }
    for site, cfg in custom_cfg.items():
        if not cfg.get("enabled"):
            continue
        for term in config.get("search_terms", [])[:4]:
            url = cfg["base_url"].format(keyword=term.replace(" ", "+"))
            log.info(f"  Custom [{site}] → '{term}'")
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code != 200:
                    log.warning(f"    ✗ HTTP {resp.status_code}")
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                found = _parse_job_cards(soup, site, term)
                jobs.extend(found)
                log.info(f"    ✓ {len(found)} risultati")
            except Exception as e:
                log.warning(f"    ✗ {e}")
    return jobs

def _parse_job_cards(soup, site_name, search_term):
    jobs = []
    for sel in ["article", ".job-card", ".job-item", "[class*='job']", ".search-result"]:
        cards = soup.select(sel)
        if len(cards) >= 2:
            break
    else:
        cards = []

    for card in cards[:30]:
        title_el = card.select_one("h2, h3, h4, [class*='title']")
        title = title_el.get_text(strip=True) if title_el else None
        if not title:
            continue
        company_el = card.select_one("[class*='company'], [class*='employer']")
        location_el = card.select_one("[class*='location'], [class*='city']")
        link_el = card.select_one("a[href]")
        desc = card.get_text(strip=True)[:500]

        if not is_relevant_job(title, desc):
            continue

        jobs.append({
            "title": title,
            "company": company_el.get_text(strip=True) if company_el else site_name.capitalize(),
            "location": location_el.get_text(strip=True) if location_el else "Roma",
            "channel": site_name.capitalize(),
            "url": link_el.get("href", "") if link_el else "",
            "search_term": search_term,
            "description": desc,
        })
    return jobs

# === FILTRO RILEVANZA POST-SCRAPING ===

def filter_relevant(df):
    if df.empty:
        return df, 0
    mask = df.apply(lambda row: is_relevant_job(row.get("title", ""), row.get("description", "")), axis=1)
    filtered = df[mask].reset_index(drop=True)
    removed = len(df) - len(filtered)
    log.info(f"  Filtro rilevanza: {len(df)} → {len(filtered)} (scartati {removed} non pertinenti)")
    if removed > 0:
        for _, row in df[~mask].head(10).iterrows():
            log.info(f"    ✗ Scartato: \"{row.get('title', '?')}\"")
    return filtered, removed

# === DEDUPLICAZIONE ===

def deduplicate(df, threshold=85):
    if df.empty:
        return df
    seen, keep = [], []
    for _, row in df.iterrows():
        t = str(row.get("title", "")).lower().strip()
        c = str(row.get("company", "")).lower().strip()
        dup = False
        for st, sc in seen:
            score = fuzz.ratio(t, st)
            if c and sc and fuzz.ratio(c, sc) > 80:
                score += 10
            if score >= threshold:
                dup = True
                break
        seen.append((t, c))
        keep.append(not dup)
    orig = len(df)
    result = df[keep].reset_index(drop=True)
    log.info(f"  Dedup: {orig} → {len(result)} (rimossi {orig - len(result)})")
    return result

# === STORICO ===

def find_new(conn, df):
    if df.empty:
        return df
    existing = {r[0] for r in conn.execute("SELECT id FROM jobs")}
    mask = [job_id(row.get("title",""), row.get("company",""), row.get("job_url", row.get("url",""))) not in existing
            for _, row in df.iterrows()]
    new = df[mask].reset_index(drop=True)
    log.info(f"  Nuove: {len(new)} su {len(df)} totali")
    return new

def save_to_db(conn, df):
    now = datetime.now().isoformat()
    n = 0
    for _, row in df.iterrows():
        jid = job_id(row.get("title",""), row.get("company",""), row.get("job_url", row.get("url","")))
        dp = row.get("date_posted", "")
        if pd.isna(dp) or str(dp) in ("None", "NaT", "nan", ""):
            dp = ""
        else:
            dp = str(dp)[:10]
        try:
            conn.execute(
                "INSERT OR IGNORE INTO jobs (id,title,company,location,channel,url,description,"
                "salary_min,salary_max,job_type,date_posted,date_scraped,search_term,is_new) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                (jid, row.get("title",""), row.get("company",""),
                 str(row.get("location", row.get("city","Roma"))),
                 str(row.get("site", row.get("channel",""))),
                 row.get("job_url", row.get("url","")),
                 str(row.get("description",""))[:2000],
                 row.get("min_amount"), row.get("max_amount"),
                 row.get("job_type",""), dp, now, row.get("search_term","")))
            n += 1
        except Exception as e:
            log.warning(f"  DB error: {e}")
    conn.commit()
    return n

# === REPORT EXCEL ===

def generate_excel(conn):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    today = datetime.now().strftime("%Y-%m-%d")
    path = OUTPUT_DIR / f"TA_Monitor_Roma_{today}.xlsx"
    wb = Workbook()
    hfont = Font(bold=True, color="FFFFFF", size=11)
    hfill = PatternFill("solid", fgColor="4A00E0")

    ws = wb.active
    ws.title = "Nuove Posizioni"
    headers = ["Titolo", "Azienda", "Canale", "Luogo", "Tipo", "Salary Min", "Salary Max", "URL", "Data"]
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font, cell.fill = hfont, hfill
        cell.alignment = Alignment(horizontal="center")

    cutoff = (datetime.now() - timedelta(hours=72)).isoformat()
    rows = conn.execute(
        "SELECT title, company, channel, location, job_type, salary_min, salary_max, url, date_posted "
        "FROM jobs WHERE date_scraped > ? ORDER BY date_scraped DESC", (cutoff,)
    ).fetchall()

    for r, row in enumerate(rows, 2):
        for c, val in enumerate(row, 1):
            display = val if val and str(val) not in ("None", "nan", "NaT", "") else ""
            cell = ws.cell(row=r, column=c, value=display)
            if c == 8 and val:
                cell.font = Font(color="0066CC", underline="single")

    for c, w in enumerate([45, 30, 12, 25, 12, 12, 12, 55, 12], 1):
        ws.column_dimensions[chr(64 + c)].width = w
    if rows:
        ws.auto_filter.ref = f"A1:I{len(rows) + 1}"
    ws.freeze_panes = "A2"

    ws2 = wb.create_sheet("Per Canale")
    for c, h in enumerate(["Canale", "Questa Scan", "Totale"], 1):
        cell = ws2.cell(row=1, column=c, value=h)
        cell.font, cell.fill = hfont, hfill
    for r, row in enumerate(conn.execute(
        "SELECT channel, SUM(CASE WHEN date_scraped > ? THEN 1 ELSE 0 END), COUNT(*) "
        "FROM jobs GROUP BY channel ORDER BY 2 DESC", (cutoff,)), 2):
        for c, val in enumerate(row, 1):
            ws2.cell(row=r, column=c, value=val or "")

    ws3 = wb.create_sheet("Trend 7gg")
    for c, h in enumerate(["Data", "Posizioni"], 1):
        cell = ws3.cell(row=1, column=c, value=h)
        cell.font, cell.fill = hfont, hfill
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    for r, row in enumerate(conn.execute(
        "SELECT DATE(date_scraped), COUNT(*) FROM jobs WHERE date_scraped > ? "
        "GROUP BY DATE(date_scraped) ORDER BY 1 DESC", (week_ago,)), 2):
        for c, val in enumerate(row, 1):
            ws3.cell(row=r, column=c, value=val)

    wb.save(path)
    log.info(f"  Excel: {path}")
    return path

# === SUMMARY ===

def write_summary(total, new, dupes, filtered, channels):
    s = {"date": datetime.now().isoformat(), "total_raw": total, "filtered_irrelevant": filtered,
         "new_jobs": new, "duplicates_removed": dupes, "channels": channels}
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(s, f, indent=2)
    return s

# === MAIN ===

def main():
    parser = argparse.ArgumentParser(description="TA Monitor — HR/TA Roma")
    parser.add_argument("--hours-old", type=int, default=72)
    parser.add_argument("--search-terms", type=str, default="")
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("TA MONITOR v2.1 — Avvio scansione")
    log.info(f"  Finestra: ultime {args.hours_old}h")
    log.info("=" * 55)

    config = load_config()
    if args.search_terms:
        config["search_terms"] = [t.strip() for t in args.search_terms.split(",")]

    conn = init_db()

    log.info("▸ Fase 1: Scraping JobSpy")
    jobspy_df = scrape_jobspy(config, args.hours_old)

    log.info("▸ Fase 2: Scraping agenzie")
    custom_jobs = scrape_custom(config)

    log.info("▸ Fase 3: Merge")
    if custom_jobs:
        custom_df = pd.DataFrame(custom_jobs)
        all_df = pd.concat([jobspy_df, custom_df], ignore_index=True) if not jobspy_df.empty else custom_df
    else:
        all_df = jobspy_df
    total_raw = len(all_df)
    log.info(f"  Totale grezzo: {total_raw}")

    if all_df.empty:
        log.warning("Nessun risultato. Fine.")
        write_summary(0, 0, 0, 0, [])
        conn.close()
        return

    log.info("▸ Fase 4: Filtro rilevanza HR/TA")
    relevant_df, filtered_count = filter_relevant(all_df)
    if relevant_df.empty:
        log.warning("Nessuna posizione rilevante dopo il filtro.")
        write_summary(total_raw, 0, 0, filtered_count, [])
        conn.close()
        return

    log.info("▸ Fase 5: Deduplicazione")
    deduped = deduplicate(relevant_df)

    log.info("▸ Fase 6: Confronto storico")
    new_df = find_new(conn, deduped)

    log.info("▸ Fase 7: Salvataggio")
    save_to_db(conn, deduped)

    log.info("▸ Fase 8: Report")
    generate_excel(conn)
    today = datetime.now().strftime("%Y-%m-%d")
    deduped.to_csv(OUTPUT_DIR / f"ta_jobs_{today}.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    deduped.to_json(OUTPUT_DIR / f"ta_jobs_{today}.json", orient="records", force_ascii=False, indent=2)

    channels = list(deduped["site"].unique()) if "site" in deduped.columns else []
    write_summary(total_raw, len(new_df), len(relevant_df) - len(deduped), filtered_count, channels)

    conn.execute(
        "INSERT INTO runs (timestamp,total_found,new_found,duplicates_removed,filtered_irrelevant,status) "
        "VALUES (?,?,?,?,?,?)",
        (datetime.now().isoformat(), total_raw, len(new_df),
         len(relevant_df) - len(deduped), filtered_count, "success"))
    conn.commit()
    conn.close()

    log.info("=" * 55)
    log.info(f"✓ Grezzo: {total_raw} | Scartati: {filtered_count} | Rilevanti: {len(deduped)} | Nuove: {len(new_df)}")
    log.info("=" * 55)

if __name__ == "__main__":
    main()
