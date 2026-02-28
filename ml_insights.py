#!/usr/bin/env python3
"""
TA Monitor — ML Insights Engine v1.0

Analisi intelligente degli annunci di lavoro:
1. Scoring: punteggio rilevanza 0-100 per ogni annuncio
2. Classificazione: migliora filtri suggerendo nuovi pattern/blacklist
3. Raccomandazione keyword: suggerisce nuove keyword di ricerca
4. Anomaly detection: segnala cambiamenti insoliti nel mercato

Uso:
    python ml_insights.py --profile hr-roma
    python ml_insights.py --profile all
"""

import argparse
import json
import logging
import re
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# === CONFIG ===
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"

log = logging.getLogger("ml_insights")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [ML] %(message)s", datefmt="%H:%M:%S")

# Stop words italiane + inglesi comuni
STOP_WORDS = set("""
di del della delle dei e a in per la il lo le un una the and or for to at of with on is an
da al alla con su che non si è sono ha come più suo sua suoi nei nel tra fra anche questo
questa questi queste stato essere molto può tutto tutti quello quella quelli quelle
""".split())


def load_profiles():
    """Carica profiles.json."""
    fpath = BASE_DIR / "profiles.json"
    if fpath.exists():
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_jobs_for_profile(profile_id):
    """Carica tutti i job JSON disponibili per un profilo."""
    # Try profile subdir first, then root
    dirs_to_try = [OUTPUT_DIR / profile_id, OUTPUT_DIR]
    all_jobs = []

    for job_dir in dirs_to_try:
        if not job_dir.exists():
            continue
        json_files = sorted(job_dir.glob("ta_jobs_*.json"))
        if json_files:
            for jf in json_files[-15:]:  # Last 15 files max
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        jobs = json.load(f)
                    scan_date = jf.stem.replace("ta_jobs_", "")
                    for j in jobs:
                        j["_scan_date"] = scan_date
                    all_jobs.extend(jobs)
                except Exception as e:
                    log.warning(f"  Skip {jf.name}: {e}")
            break  # Found files, stop looking

    return all_jobs


def load_all_raw_jobs(profile_id):
    """Carica anche i job grezzi (pre-filtro) se disponibili dal DB."""
    import sqlite3
    db_path = DATA_DIR / "ta_monitor.db" if profile_id in ("default", "hr-roma") else DATA_DIR / f"{profile_id}.db"
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM jobs ORDER BY date_scraped DESC LIMIT 2000").fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        log.warning(f"  DB read failed: {e}")
        return []


# ============================================================
# 1. SCORING — Punteggio rilevanza 0-100
# ============================================================
def compute_scores(jobs, profile):
    """Calcola score di rilevanza per ogni annuncio usando TF-IDF + pattern matching."""
    if not jobs:
        return []

    filters = profile.get("filters", {})
    relevant_patterns = [p["pattern"] if isinstance(p, dict) else p for p in filters.get("relevant_patterns", [])]
    blacklist_patterns = [p["pattern"] if isinstance(p, dict) else p for p in filters.get("blacklist", [])]
    broad_keywords = filters.get("broad_keywords", [])
    search_terms = filters.get("search_terms", [])

    # Build reference corpus from search terms + broad keywords
    reference_text = " ".join(search_terms + broad_keywords)

    # Build title corpus
    titles = [j.get("title", "") or "" for j in jobs]
    descriptions = [j.get("description", "") or "" for j in jobs]
    combined = [f"{t} {d[:200]}" for t, d in zip(titles, descriptions)]

    # TF-IDF similarity with reference
    try:
        corpus = [reference_text] + combined
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words=list(STOP_WORDS),
            min_df=1,
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    except Exception:
        similarities = np.zeros(len(jobs))

    scores = []
    for i, job in enumerate(jobs):
        title = (job.get("title", "") or "").strip()
        desc = (job.get("description", "") or "")[:500]
        score = 0.0

        # Component 1: TF-IDF similarity (0-40 points)
        score += similarities[i] * 40

        # Component 2: Pattern match (0-35 points)
        pattern_score = 0
        for pat in relevant_patterns:
            try:
                if re.search(pat, title, re.IGNORECASE):
                    pattern_score = 35
                    break
                elif re.search(pat, desc, re.IGNORECASE):
                    pattern_score = 20
                    break
            except re.error:
                pass
        score += pattern_score

        # Component 3: Broad keyword match (0-15 points)
        title_lower = title.lower()
        kw_matches = sum(1 for kw in broad_keywords if kw.lower() in title_lower)
        score += min(kw_matches * 5, 15)

        # Component 4: Blacklist penalty (-30 points)
        for pat in blacklist_patterns:
            try:
                if re.search(pat, title, re.IGNORECASE):
                    score -= 30
                    break
            except re.error:
                pass

        # Component 5: Freshness bonus (0-10 points)
        date_posted = job.get("date_posted")
        if date_posted:
            try:
                age_days = _days_old(date_posted)
                if age_days is not None and age_days <= 3:
                    score += 10
                elif age_days is not None and age_days <= 7:
                    score += 5
            except Exception:
                pass

        # Clamp to 0-100
        final_score = max(0, min(100, round(score)))
        job["_ml_score"] = final_score
        scores.append(final_score)

    return scores


# ============================================================
# 2. CLASSIFICAZIONE — Suggerimenti per migliorare i filtri
# ============================================================
def suggest_filter_improvements(jobs, profile):
    """Analizza i job per suggerire miglioramenti ai filtri."""
    filters = profile.get("filters", {})
    blacklist_patterns = [p["pattern"] if isinstance(p, dict) else p for p in filters.get("blacklist", [])]
    relevant_patterns = [p["pattern"] if isinstance(p, dict) else p for p in filters.get("relevant_patterns", [])]

    suggestions = {"add_relevant": [], "add_blacklist": [], "review": []}

    # Find high-scored jobs that DON'T match any relevant pattern
    # These might need new patterns
    for job in jobs:
        title = (job.get("title", "") or "").strip()
        score = job.get("_ml_score", 0)

        if score >= 60:
            matched = any(
                _safe_search(pat, title) for pat in relevant_patterns
            )
            if not matched and title:
                suggestions["add_relevant"].append({
                    "title": title,
                    "score": score,
                    "reason": "Alto score TF-IDF ma nessun pattern match"
                })

    # Find low-scored jobs that ARE in results (possible false positives)
    for job in jobs:
        title = (job.get("title", "") or "").strip()
        score = job.get("_ml_score", 0)

        if score < 25 and title:
            # Check if it matches a relevant pattern (false positive)
            matched_pat = None
            for pat in relevant_patterns:
                if _safe_search(pat, title):
                    matched_pat = pat
                    break
            if matched_pat:
                suggestions["review"].append({
                    "title": title,
                    "score": score,
                    "pattern": matched_pat,
                    "reason": "Basso score ma match con pattern — possibile falso positivo"
                })

    # Find common words in low-score titles for blacklist suggestions
    low_score_titles = [j.get("title", "") for j in jobs if j.get("_ml_score", 0) < 20]
    if low_score_titles:
        word_freq = _extract_significant_words(low_score_titles)
        high_score_words = set()
        high_titles = [j.get("title", "") for j in jobs if j.get("_ml_score", 0) >= 60]
        if high_titles:
            high_score_words = set(_extract_significant_words(high_titles).keys())

        for word, count in word_freq.most_common(10):
            if word not in high_score_words and count >= 3:
                # Check if already in blacklist
                already_blocked = any(_safe_search(pat, word) for pat in blacklist_patterns)
                if not already_blocked:
                    suggestions["add_blacklist"].append({
                        "word": word,
                        "count": count,
                        "reason": f"Appare {count}x in annunci a basso score"
                    })

    # Limit suggestions
    suggestions["add_relevant"] = suggestions["add_relevant"][:5]
    suggestions["add_blacklist"] = suggestions["add_blacklist"][:5]
    suggestions["review"] = suggestions["review"][:5]

    return suggestions


# ============================================================
# 3. RACCOMANDAZIONE KEYWORD
# ============================================================
def recommend_keywords(jobs, profile):
    """Suggerisce nuove keyword basandosi sui titoli ad alto score."""
    filters = profile.get("filters", {})
    current_keywords = set(kw.lower() for kw in filters.get("search_terms", []))
    broad_keywords = set(kw.lower() for kw in filters.get("broad_keywords", []))
    all_known = current_keywords | broad_keywords

    # Get titles of high-score jobs
    good_titles = [j.get("title", "") for j in jobs if j.get("_ml_score", 0) >= 50]
    if not good_titles:
        return []

    # Extract bigrams and trigrams
    bigram_counter = Counter()
    for title in good_titles:
        words = _tokenize(title)
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                if not any(w in STOP_WORDS for w in words[i:i + n]):
                    bigram_counter[ngram] += 1

    # Also extract single significant words
    word_counter = _extract_significant_words(good_titles)

    recommendations = []

    # Bigrams/trigrams not in current keywords
    for ngram, count in bigram_counter.most_common(30):
        if count >= 2 and ngram.lower() not in all_known:
            # Check it's not too similar to existing keywords
            if not any(ngram.lower() in kw or kw in ngram.lower() for kw in all_known):
                recommendations.append({
                    "keyword": ngram,
                    "frequency": count,
                    "source": "bigram/trigram da titoli rilevanti",
                    "confidence": min(count * 15, 90)
                })

    # Single words with high frequency
    for word, count in word_counter.most_common(20):
        if count >= 3 and word not in all_known and len(word) > 3:
            if not any(word in kw or kw in word for kw in all_known):
                recommendations.append({
                    "keyword": word,
                    "frequency": count,
                    "source": "parola frequente in titoli rilevanti",
                    "confidence": min(count * 10, 80)
                })

    # Sort by confidence, deduplicate
    seen = set()
    unique_recs = []
    for r in sorted(recommendations, key=lambda x: x["confidence"], reverse=True):
        if r["keyword"].lower() not in seen:
            seen.add(r["keyword"].lower())
            unique_recs.append(r)

    return unique_recs[:10]


# ============================================================
# 4. ANOMALY DETECTION
# ============================================================
def detect_anomalies(jobs, profile):
    """Rileva cambiamenti insoliti nel mercato del lavoro."""
    anomalies = []

    if len(jobs) < 10:
        return anomalies

    # Parse dates
    dated_jobs = []
    for j in jobs:
        d = _parse_date(j.get("date_posted"))
        if d:
            dated_jobs.append((d, j))
    dated_jobs.sort(key=lambda x: x[0])

    if not dated_jobs:
        return anomalies

    now = datetime.now()

    # --- Volume anomaly ---
    # Compare this week vs previous weeks
    weekly_counts = defaultdict(int)
    for d, j in dated_jobs:
        week_key = f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"
        weekly_counts[week_key] += 1

    weeks = sorted(weekly_counts.keys())
    if len(weeks) >= 3:
        recent = weekly_counts[weeks[-1]]
        prev_avg = np.mean([weekly_counts[w] for w in weeks[:-1]])
        if prev_avg > 0:
            change_pct = ((recent - prev_avg) / prev_avg) * 100
            std_dev = np.std([weekly_counts[w] for w in weeks[:-1]])
            z_score = (recent - prev_avg) / std_dev if std_dev > 0 else 0

            if abs(z_score) > 1.5:
                anomalies.append({
                    "type": "volume_spike" if change_pct > 0 else "volume_drop",
                    "severity": "high" if abs(z_score) > 2 else "medium",
                    "message": f"Volume {'in forte crescita' if change_pct > 0 else 'in calo'}: {recent} annunci questa settimana vs media {prev_avg:.0f} ({change_pct:+.0f}%)",
                    "z_score": round(z_score, 2),
                    "data": {"current": recent, "avg": round(prev_avg, 1), "change_pct": round(change_pct, 1)}
                })

    # --- New company surge ---
    # Companies appearing for the first time this week with multiple postings
    this_week = now - timedelta(days=7)
    recent_companies = Counter()
    old_companies = set()

    for d, j in dated_jobs:
        company = (j.get("company") or "").strip()
        if not company:
            continue
        if d >= this_week:
            recent_companies[company] += 1
        else:
            old_companies.add(company.lower())

    for company, count in recent_companies.most_common(5):
        if company.lower() not in old_companies and count >= 2:
            anomalies.append({
                "type": "new_company_surge",
                "severity": "medium",
                "message": f"🆕 {company} è apparsa questa settimana con {count} posizioni (mai vista prima)",
                "data": {"company": company, "count": count}
            })

    # --- Channel shift ---
    recent_channels = Counter()
    old_channels = Counter()
    for d, j in dated_jobs:
        ch = (j.get("site") or "").lower()
        if not ch:
            continue
        if d >= this_week:
            recent_channels[ch] += 1
        else:
            old_channels[ch] += 1

    total_old = sum(old_channels.values()) or 1
    total_recent = sum(recent_channels.values()) or 1
    for ch in set(list(recent_channels.keys()) + list(old_channels.keys())):
        old_pct = (old_channels.get(ch, 0) / total_old) * 100
        new_pct = (recent_channels.get(ch, 0) / total_recent) * 100
        shift = new_pct - old_pct
        if abs(shift) > 15:
            anomalies.append({
                "type": "channel_shift",
                "severity": "low",
                "message": f"📡 {ch}: quota passata da {old_pct:.0f}% a {new_pct:.0f}% ({shift:+.0f}pp)",
                "data": {"channel": ch, "old_pct": round(old_pct, 1), "new_pct": round(new_pct, 1)}
            })

    # --- Role type clustering ---
    # Use TF-IDF + KMeans to find clusters in titles
    titles = [j.get("title", "") for _, j in dated_jobs if j.get("title")]
    if len(titles) >= 20:
        try:
            vectorizer = TfidfVectorizer(max_features=200, stop_words=list(STOP_WORDS), min_df=2)
            X = vectorizer.fit_transform(titles)
            n_clusters = min(5, len(titles) // 5)
            if n_clusters >= 2:
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = km.fit_predict(X)

                # Find the dominant terms per cluster
                feature_names = vectorizer.get_feature_names_out()
                clusters = []
                for c in range(n_clusters):
                    center = km.cluster_centers_[c]
                    top_indices = center.argsort()[-3:][::-1]
                    top_terms = [feature_names[i] for i in top_indices]
                    cluster_size = int((labels == c).sum())
                    clusters.append({
                        "id": c,
                        "size": cluster_size,
                        "top_terms": top_terms,
                        "pct": round(cluster_size / len(titles) * 100, 1)
                    })

                clusters.sort(key=lambda x: x["size"], reverse=True)
                anomalies.append({
                    "type": "role_clusters",
                    "severity": "info",
                    "message": f"🔬 Identificati {n_clusters} cluster di ruoli nel mercato",
                    "data": {"clusters": clusters}
                })
        except Exception as e:
            log.warning(f"  Clustering skip: {e}")

    return anomalies


# ============================================================
# HELPERS
# ============================================================
def _tokenize(text):
    """Tokenizza un testo in parole."""
    return [w.lower() for w in re.findall(r'[a-zàèéìòùA-Z]{2,}', text.lower()) if w.lower() not in STOP_WORDS]


def _extract_significant_words(titles):
    """Estrae parole significative dai titoli."""
    counter = Counter()
    for title in titles:
        for word in _tokenize(title):
            if len(word) > 2:
                counter[word] += 1
    return counter


def _safe_search(pattern, text):
    """Regex search con gestione errori."""
    try:
        return bool(re.search(pattern, text, re.IGNORECASE))
    except re.error:
        return False


def _days_old(date_val):
    """Calcola l'età in giorni di una data."""
    d = _parse_date(date_val)
    if d:
        return (datetime.now() - d).days
    return None


def _parse_date(raw):
    """Parse vari formati di data."""
    if not raw or raw in ("None", "NaT", "nan", ""):
        return None
    try:
        num = float(raw)
        if num > 1e12:
            return datetime.fromtimestamp(num / 1000)
        elif num > 1e9:
            return datetime.fromtimestamp(num)
    except (ValueError, TypeError, OSError):
        pass
    if isinstance(raw, str) and len(raw) >= 10:
        try:
            return datetime.strptime(raw[:10], "%Y-%m-%d")
        except ValueError:
            pass
    return None


# ============================================================
# MAIN — Generate insights
# ============================================================
def generate_insights(profile_id, profile):
    """Genera tutti gli insights ML per un profilo."""
    log.info(f"{'='*50}")
    log.info(f"ML INSIGHTS — Profilo: {profile.get('name', profile_id)}")
    log.info(f"{'='*50}")

    # Load jobs
    jobs = load_jobs_for_profile(profile_id)
    if not jobs:
        log.warning("Nessun job trovato, skip ML")
        return None

    # Deduplicate
    seen = set()
    unique = []
    for j in jobs:
        key = f"{(j.get('title','') or '').lower()}|{(j.get('company','') or '').lower()}"
        if key not in seen:
            seen.add(key)
            unique.append(j)
    jobs = unique
    log.info(f"  Jobs caricati: {len(jobs)} (deduplicati)")

    # 1. Scoring
    log.info("▸ Fase 1: Scoring rilevanza")
    scores = compute_scores(jobs, profile)
    avg_score = np.mean(scores) if scores else 0
    log.info(f"  Score medio: {avg_score:.1f} | Range: {min(scores)}-{max(scores)}")

    # Score distribution
    score_dist = {
        "excellent": len([s for s in scores if s >= 80]),
        "good": len([s for s in scores if 60 <= s < 80]),
        "medium": len([s for s in scores if 40 <= s < 60]),
        "low": len([s for s in scores if 20 <= s < 40]),
        "irrelevant": len([s for s in scores if s < 20]),
    }
    log.info(f"  Distribuzione: {score_dist}")

    # 2. Classification suggestions
    log.info("▸ Fase 2: Suggerimenti filtri")
    filter_suggestions = suggest_filter_improvements(jobs, profile)
    log.info(f"  Nuovi pattern: {len(filter_suggestions['add_relevant'])} | Blacklist: {len(filter_suggestions['add_blacklist'])} | Review: {len(filter_suggestions['review'])}")

    # 3. Keyword recommendations
    log.info("▸ Fase 3: Raccomandazione keyword")
    keyword_recs = recommend_keywords(jobs, profile)
    log.info(f"  Keyword suggerite: {len(keyword_recs)}")
    for r in keyword_recs[:3]:
        log.info(f"    → \"{r['keyword']}\" (freq: {r['frequency']}, conf: {r['confidence']}%)")

    # 4. Anomaly detection
    log.info("▸ Fase 4: Anomaly detection")
    anomalies = detect_anomalies(jobs, profile)
    log.info(f"  Anomalie rilevate: {len(anomalies)}")
    for a in anomalies:
        log.info(f"    [{a['severity'].upper()}] {a['message']}")

    # Build output
    # Top scored jobs
    top_jobs = sorted(jobs, key=lambda j: j.get("_ml_score", 0), reverse=True)[:20]
    top_jobs_clean = [{
        "title": j.get("title", ""),
        "company": j.get("company", ""),
        "score": j.get("_ml_score", 0),
        "site": j.get("site", ""),
        "date_posted": j.get("date_posted", ""),
        "url": j.get("job_url") or j.get("job_url_direct") or j.get("url") or ""
    } for j in top_jobs]

    insights = {
        "generated_at": datetime.now().isoformat(),
        "profile_id": profile_id,
        "profile_name": profile.get("name", profile_id),
        "total_jobs_analyzed": len(jobs),
        "scoring": {
            "average": round(avg_score, 1),
            "distribution": score_dist,
            "top_jobs": top_jobs_clean
        },
        "filter_suggestions": filter_suggestions,
        "keyword_recommendations": keyword_recs,
        "anomalies": anomalies
    }

    # Save
    out_dir = OUTPUT_DIR / profile_id if profile_id not in ("default",) else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ml_insights.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"✓ Salvato: {out_path}")

    return insights


def main():
    parser = argparse.ArgumentParser(description="TA Monitor ML Insights")
    parser.add_argument("--profile", type=str, default="all")
    args = parser.parse_args()

    profiles_data = load_profiles()
    if not profiles_data or "profiles" not in profiles_data:
        log.error("Nessun profiles.json trovato")
        return

    profiles = profiles_data["profiles"]

    if args.profile == "all":
        for pid, profile in profiles.items():
            try:
                generate_insights(pid, profile)
            except Exception as e:
                log.error(f"Errore profilo {pid}: {e}")
    else:
        if args.profile not in profiles:
            log.error(f"Profilo '{args.profile}' non trovato")
            return
        generate_insights(args.profile, profiles[args.profile])


if __name__ == "__main__":
    main()
