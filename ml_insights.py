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


def load_feedback():
    """Carica feedback.json con i voti degli utenti."""
    fpath = BASE_DIR / "feedback.json"
    if fpath.exists():
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


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
def compute_scores(jobs, profile, feedback=None):
    """Calcola score di rilevanza per ogni annuncio usando TF-IDF + pattern matching + feedback utente."""
    if not jobs:
        return []

    # Aggregate feedback: build sets of liked/disliked job keys
    liked_keys = set()
    disliked_keys = set()
    liked_words = Counter()
    disliked_words = Counter()
    if feedback:
        for user, user_fb in feedback.items():
            for key, fb in user_fb.items():
                if isinstance(fb, dict):
                    vote = fb.get("vote", "")
                    title = fb.get("title", "")
                else:
                    continue
                if vote == "up":
                    liked_keys.add(key)
                    for w in _tokenize(title):
                        liked_words[w] += 1
                elif vote == "down":
                    disliked_keys.add(key)
                    for w in _tokenize(title):
                        disliked_words[w] += 1

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

        # Component 6: Feedback learning (-15 to +15 points)
        job_key = f"{title_lower}|{(job.get('company', '') or '').lower().strip()}"
        if job_key in liked_keys:
            score += 15  # Direct match with liked job
        elif job_key in disliked_keys:
            score -= 15  # Direct match with disliked job
        else:
            # Indirect: boost if title words match liked patterns
            title_words = set(_tokenize(title))
            liked_overlap = sum(liked_words.get(w, 0) for w in title_words)
            disliked_overlap = sum(disliked_words.get(w, 0) for w in title_words)
            if liked_overlap > disliked_overlap:
                score += min(10, liked_overlap * 2)
            elif disliked_overlap > liked_overlap:
                score -= min(10, disliked_overlap * 2)

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
def detect_anomalies(jobs, profile, profile_id):
    """Rileva cambiamenti insoliti nel mercato del lavoro.
    
    Usa _scan_date (data del file) per confronto storico, NON date_posted,
    perché ogni scan ri-scarica gli stessi annunci e date_posted è la data
    dell'annuncio originale, non di quando l'abbiamo visto per la prima volta.
    """
    anomalies = []

    if len(jobs) < 10:
        return anomalies

    now = datetime.now()

    # --- Group jobs by scan date (from filename) ---
    by_scan = defaultdict(list)
    for j in jobs:
        sd = j.get("_scan_date", "")
        if sd:
            by_scan[sd].append(j)

    scan_dates = sorted(by_scan.keys())
    if len(scan_dates) < 2:
        # Not enough history for comparison
        anomalies.append({
            "type": "info",
            "severity": "info",
            "message": f"📊 {len(jobs)} annunci analizzati su {len(scan_dates)} scan. Servono più scan per confronti storici.",
            "data": {}
        })

    # --- Volume trend across scans ---
    if len(scan_dates) >= 3:
        # Count UNIQUE jobs per scan (by title+company)
        scan_unique_counts = {}
        for sd, sj in by_scan.items():
            seen = set()
            for j in sj:
                key = f"{(j.get('title','') or '').lower()}|{(j.get('company','') or '').lower()}"
                seen.add(key)
            scan_unique_counts[sd] = len(seen)

        counts = [scan_unique_counts[sd] for sd in scan_dates]
        latest = counts[-1]
        prev_avg = np.mean(counts[:-1])
        if prev_avg > 0:
            change_pct = ((latest - prev_avg) / prev_avg) * 100
            if abs(change_pct) > 30:
                anomalies.append({
                    "type": "volume_spike" if change_pct > 0 else "volume_drop",
                    "severity": "high" if abs(change_pct) > 50 else "medium",
                    "message": f"Volume {'in crescita' if change_pct > 0 else 'in calo'}: {latest} annunci ultimo scan vs media {prev_avg:.0f} ({change_pct:+.0f}%)",
                    "data": {"current": latest, "avg": round(prev_avg, 1), "change_pct": round(change_pct, 1)}
                })

    # --- Truly new companies (compare latest scan vs all previous) ---
    if len(scan_dates) >= 2:
        latest_scan = scan_dates[-1]
        prev_scans = scan_dates[:-1]

        old_companies = set()
        for sd in prev_scans:
            for j in by_scan[sd]:
                c = (j.get("company") or "").strip().lower()
                if c:
                    old_companies.add(c)

        new_companies = Counter()
        for j in by_scan[latest_scan]:
            c = (j.get("company") or "").strip()
            if c and c.lower() not in old_companies:
                new_companies[c] += 1

        for company, count in new_companies.most_common(5):
            if count >= 2:
                anomalies.append({
                    "type": "new_company_surge",
                    "severity": "medium",
                    "message": f"🆕 {company}: {count} posizioni, non presente negli scan precedenti",
                    "data": {"company": company, "count": count}
                })

        total_new = sum(new_companies.values())
        total_latest = len(by_scan[latest_scan])
        if total_latest > 0 and total_new > 0:
            new_pct = round(total_new / total_latest * 100, 1)
            anomalies.append({
                "type": "new_jobs_ratio",
                "severity": "info",
                "message": f"📊 Ultimo scan: {total_new}/{total_latest} annunci da aziende nuove ({new_pct}%)",
                "data": {"new": total_new, "total": total_latest, "pct": new_pct}
            })

    # --- Channel distribution (latest scan only, no false comparison) ---
    if scan_dates:
        latest_jobs = by_scan[scan_dates[-1]]
        ch_counts = Counter()
        for j in latest_jobs:
            ch = (j.get("site") or "").lower()
            if ch:
                ch_counts[ch] += 1
        total = sum(ch_counts.values()) or 1

        # Only report if there's a previous scan to compare
        if len(scan_dates) >= 2:
            prev_jobs = by_scan[scan_dates[-2]]
            prev_ch = Counter()
            for j in prev_jobs:
                ch = (j.get("site") or "").lower()
                if ch:
                    prev_ch[ch] += 1
            prev_total = sum(prev_ch.values()) or 1

            for ch in set(list(ch_counts.keys()) + list(prev_ch.keys())):
                old_pct = (prev_ch.get(ch, 0) / prev_total) * 100
                new_pct = (ch_counts.get(ch, 0) / total) * 100
                shift = new_pct - old_pct
                if abs(shift) > 15:
                    anomalies.append({
                        "type": "channel_shift",
                        "severity": "low",
                        "message": f"📡 {ch}: quota passata da {old_pct:.0f}% a {new_pct:.0f}% ({shift:+.0f}pp)",
                        "data": {"channel": ch, "old_pct": round(old_pct, 1), "new_pct": round(new_pct, 1)}
                    })

    # --- Role type clustering ---
    titles = [j.get("title", "") for j in jobs if j.get("title")]
    if len(titles) >= 20:
        try:
            vectorizer = TfidfVectorizer(max_features=200, stop_words=list(STOP_WORDS), min_df=2)
            X = vectorizer.fit_transform(titles)
            n_clusters = min(5, len(titles) // 5)
            if n_clusters >= 2:
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = km.fit_predict(X)

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

    # Load user feedback
    feedback = load_feedback()
    fb_count = sum(len(v) for v in feedback.values()) if feedback else 0
    log.info(f"  Feedback utenti caricati: {fb_count} voti")

    # 1. Scoring
    log.info("▸ Fase 1: Scoring rilevanza")
    scores = compute_scores(jobs, profile, feedback)
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
    anomalies = detect_anomalies(jobs, profile, profile_id)
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
        "anomalies": anomalies,
        "feedback_stats": {
            "total_votes": fb_count,
            "users": len(feedback),
            "feedback_boost_active": fb_count > 0
        }
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