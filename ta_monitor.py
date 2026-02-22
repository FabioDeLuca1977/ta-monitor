#!/usr/bin/env python3
"""
TA Monitor — Monitoraggio Posizioni Talent Acquisition Roma
Maieutike srl — v2.0 (GitHub Actions edition)

Uso:
    python ta_monitor.py                              # Default: ultime 24h
    python ta_monitor.py --hours-old 48               # Ultime 48h
    python ta_monitor.py --search-terms "HR manager"  # Keywords custom
"""

import argparse
import csv
import hashlib
import json
import logging
import os
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

# === CONFIG ===

def load_config():
    cfg_path = BASE_DIR / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
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


# === SCRAPING: JOBSPY ===

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


# === SCRAPING: CUSTOM (Randstad, Manpower, Adecco) ===

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
        # Usa solo le prime 3 keyword per i siti custom
        for term in config.get("search_terms", [])[:3]:
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
    # Cerca job cards con selettori comuni
    for sel in ["article", ".job-card", ".job-item", "[class*='job']", ".search-result"]:
        cards = soup.select(sel)
        if len(cards) >= 2:
            break
    else:
        cards = []

    relevant = {"talent", "recruiter", "recruiting", "selezione", "hr", "risorse umane", "head hunter"}

    for card in cards[:20]:
        title_el = card.select_one("h2, h3, h4, [class*='title']")
        title = title_el.get_text(strip=True) if title_el else None
        if not title or not any(t in title.lower() for t in relevant):
            continue

        company_el = card.select_one("[class*='company'], [class*='employer']")
        location_el = card.select_one("[class*='location'], [class*='city']")
        link_el = card.select_one("a[href]")

        jobs.append({
            "title": title,
            "company": company_el.get_text(strip=True) if company_el else site_name.capitalize(),
            "location": location_el.get_text(strip=True) if location_el else "Roma",
            "channel": site_name.capitalize(),
            "url": link_el.get("href", "") if link_el else "",
            "search_term": search_term,
            "description": card.get_text(strip=True)[:500],
        })
    return jobs


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
    mask = []
    for _, row in df.iterrows():
        jid = job_id(row.get("title", ""), row.get("company", ""), row.get("job_url", row.get("url", "")))
        mask.append(jid not in existing)
    new = df[mask].reset_index(drop=True)
    log.info(f"  Nuove: {len(new)} su {len(df)} totali")
    return new


def save_to_db(conn, df):
    now = datetime.now().isoformat()
    n = 0
    for _, row in df.iterrows():
        jid = job_id(row.get("title", ""), row.get("company", ""), row.get("job_url", row.get("url", "")))
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
                 row.get("job_type",""), str(row.get("date_posted","")),
                 now, row.get("search_term","")),
            )
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

    # Sheet 1: Nuove posizioni
    ws = wb.active
    ws.title = "Nuove Posizioni"
    headers = ["Titolo", "Azienda", "Canale", "Luogo", "Tipo", "Salary Min", "Salary Max", "URL", "Data"]
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font, cell.fill = hfont, hfill
        cell.alignment = Alignment(horizontal="center")

    yesterday = (datetime.now() - timedelta(hours=24)).isoformat()
    rows = conn.execute(
        "SELECT title, company, channel, location, job_type, salary_min, salary_max, url, date_posted "
        "FROM jobs WHERE date_scraped > ? ORDER BY date_scraped DESC", (yesterday,)
    ).fetchall()

    for r, row in enumerate(rows, 2):
        for c, val in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=val or "")
            if c == 8 and val:
                cell.font = Font(color="0066CC", underline="single")

    for c in range(1, len(headers) + 1):
        ws.column_dimensions[chr(64 + c)].width = min(30, 12)
    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["H"].width = 50
    ws.auto_filter.ref = f"A1:I{len(rows) + 1}"
    ws.freeze_panes = "A2"

    # Sheet 2: Per canale
    ws2 = wb.create_sheet("Per Canale")
    for c, h in enumerate(["Canale", "Oggi", "Totale"], 1):
        cell = ws2.cell(row=1, column=c, value=h)
        cell.font, cell.fill = hfont, hfill
    for r, row in enumerate(conn.execute(
        "SELECT channel, SUM(CASE WHEN date_scraped > ? THEN 1 ELSE 0 END), COUNT(*) "
        "FROM jobs GROUP BY channel ORDER BY 2 DESC", (yesterday,)
    ), 2):
        for c, val in enumerate(row, 1):
            ws2.cell(row=r, column=c, value=val or "")

    # Sheet 3: Trend 7gg
    ws3 = wb.create_sheet("Trend 7gg")
    for c, h in enumerate(["Data", "Nuove Posizioni"], 1):
        cell = ws3.cell(row=1, column=c, value=h)
        cell.font, cell.fill = hfont, hfill
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    for r, row in enumerate(conn.execute(
        "SELECT DATE(date_scraped), COUNT(*) FROM jobs WHERE date_scraped > ? "
        "GROUP BY DATE(date_scraped) ORDER BY 1 DESC", (week_ago,)
    ), 2):
        for c, val in enumerate(row, 1):
            ws3.cell(row=r, column=c, value=val)

    wb.save(path)
    log.info(f"  Excel: {path}")
    return path


# === SUMMARY JSON (per GitHub Actions commit message) ===

def write_summary(total, new, dupes, channels):
    summary = {
        "date": datetime.now().isoformat(),
        "total_raw": total,
        "new_jobs": new,
        "duplicates_removed": dupes,
        "channels": channels,
    }
    path = OUTPUT_DIR / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# === MAIN ===

def main():
    parser = argparse.ArgumentParser(description="TA Monitor — Talent Acquisition Roma")
    parser.add_argument("--hours-old", type=int, default=24, help="Max ore anzianità annunci")
    parser.add_argument("--search-terms", type=str, default="", help="Keywords custom (comma-sep)")
    args = parser.parse_args()

    log.info("=" * 50)
    log.info("TA MONITOR — Avvio scansione")
    log.info("=" * 50)

    config = load_config()
    
    # Override da CLI
    if args.search_terms:
        config["search_terms"] = [t.strip() for t in args.search_terms.split(",")]
    
    conn = init_db()

    # 1. JobSpy
    log.info("▸ Fase 1: Scraping JobSpy")
    jobspy_df = scrape_jobspy(config, args.hours_old)

    # 2. Custom
    log.info("▸ Fase 2: Scraping agenzie")
    custom_jobs = scrape_custom(config)

    # 3. Merge
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
        write_summary(0, 0, 0, [])
        conn.close()
        return

    # 4. Dedup
    log.info("▸ Fase 4: Deduplicazione")
    deduped = deduplicate(all_df)

    # 5. Nuove
    log.info("▸ Fase 5: Confronto storico")
    new_df = find_new(conn, deduped)

    # 6. Save
    log.info("▸ Fase 6: Salvataggio")
    save_to_db(conn, deduped)

    # 7. Report
    log.info("▸ Fase 7: Report")
    generate_excel(conn)

    today = datetime.now().strftime("%Y-%m-%d")
    deduped.to_csv(OUTPUT_DIR / f"ta_jobs_{today}.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    deduped.to_json(OUTPUT_DIR / f"ta_jobs_{today}.json", orient="records", force_ascii=False, indent=2)

    # 8. Summary
    channels = list(deduped["site"].unique()) if "site" in deduped.columns else []
    summary = write_summary(total_raw, len(new_df), total_raw - len(deduped), channels)

    # 9. Run log
    conn.execute(
        "INSERT INTO runs (timestamp,total_found,new_found,duplicates_removed,status) VALUES (?,?,?,?,?)",
        (datetime.now().isoformat(), total_raw, len(new_df), total_raw - len(deduped), "success"),
    )
    conn.commit()
    conn.close()

    log.info("=" * 50)
    log.info(f"✓ Totale: {total_raw} | Dedup: {len(deduped)} | Nuove: {len(new_df)}")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
