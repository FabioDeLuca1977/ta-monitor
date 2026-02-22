# 📋 TA Monitor — Monitoraggio Talent Acquisition Roma

**Maieutike srl** — Sistema automatizzato per monitorare le posizioni aperte di Talent Acquisition su Roma e provincia, multi-canale, con report giornaliero.

## Come funziona

Ogni giorno alle **07:00 CET**, GitHub Actions esegue automaticamente lo script che:

1. 🔍 **Scrapa** posizioni da LinkedIn, Indeed, Glassdoor, Google Jobs (via JobSpy)
2. 🏢 **Scrapa** Randstad, Manpower, Adecco (scraper custom)
3. 🧹 **Deduplica** i risultati con fuzzy matching
4. 📊 **Confronta** con lo storico per identificare solo le NUOVE posizioni
5. 📁 **Genera** report Excel (3 tab) + CSV + JSON
6. 💾 **Committa** i risultati nel repo (storico versionato)
7. 📧 **Invia email** (opzionale) con riepilogo

## Quick Start

### 1. Crea il repository

```bash
# Crea repo su GitHub (privato consigliato)
gh repo create ta-monitor --private

# Clona e copia i file
git clone https://github.com/TUO_USER/ta-monitor.git
cp -r ta-monitor-github/* ta-monitor/
cd ta-monitor
git add -A
git commit -m "Initial setup"
git push
```

### 2. Verifica che funziona

Vai su **GitHub → Actions → TA Monitor - Daily Scan → Run workflow** per esecuzione manuale.

### 3. Fatto!

Il cron è già configurato. Ogni mattina troverai:
- **Report Excel** scaricabile come Artifact nella tab Actions
- **Storico** committato nella cartella `output/`
- **Database SQLite** con tutte le posizioni trovate in `data/`

## Personalizzazione

### Modificare le keywords

Edita `config.yaml`:
```yaml
search_terms:
  - "talent acquisition"
  - "la tua keyword"
```

### Esecuzione manuale con parametri custom

Dal tab **Actions** su GitHub, clicca **Run workflow** e specifica:
- `hours_old`: quante ore indietro cercare (default 24)
- `search_terms`: keywords personalizzate separate da virgola

### Attivare le notifiche email

1. In **Settings → Secrets and variables → Actions**, aggiungi:
   - `SMTP_HOST` (es. `smtp.gmail.com`)
   - `SMTP_PORT` (es. `587`)
   - `SMTP_USER` (la tua email)
   - `SMTP_PASSWORD` (app password, non la password normale)
   - `EMAIL_RECIPIENTS` (a chi inviare)

2. In **Settings → Variables → Actions**, aggiungi:
   - `ENABLE_EMAIL` = `true`

### Proxy per LinkedIn

LinkedIn limita lo scraping. Per risultati migliori, in `config.yaml`:
```yaml
proxies:
  - "user:pass@proxy1:port"
```

## Struttura

```
ta-monitor/
├── .github/workflows/
│   └── ta-scan.yml          # GitHub Actions workflow
├── ta_monitor.py             # Script principale
├── config.yaml               # Configurazione keywords e canali
├── requirements.txt          # Dipendenze Python
├── data/
│   └── ta_monitor.db         # Database SQLite (storico)
├── output/
│   ├── TA_Monitor_Roma_2026-02-22.xlsx
│   ├── ta_jobs_2026-02-22.csv
│   ├── ta_jobs_2026-02-22.json
│   └── summary.json          # Riepilogo per commit message
└── logs/
    └── ta_monitor.log
```

## Canali monitorati

| Canale | Metodo | Note |
|---|---|---|
| LinkedIn | JobSpy | Rate limiting, proxy consigliati |
| Indeed | JobSpy | Migliore, no rate limiting |
| Glassdoor | JobSpy | Dati limitati per IT |
| Google Jobs | JobSpy | Aggrega molte fonti |
| Randstad | Scraper custom | Parsing HTML |
| Manpower | Scraper custom | Parsing HTML |
| Adecco | Scraper custom | Parsing HTML |

## Costi

**Zero.** GitHub Actions offre 2000 minuti/mese gratis per repo privati. Ogni scan dura circa 2-5 minuti, quindi ~150 minuti/mese per scan giornalieri.

## Note

- InfoJobs Italia ha chiuso definitivamente a fine 2025
- Il database SQLite cresce di pochi KB al giorno
- I report Excel vengono mantenuti 90 giorni come GitHub Artifacts
- Lo storico Git permette di vedere l'evoluzione nel tempo
