#!/usr/bin/env python3
"""
TA Monitor — Email Report Personalizzato

Invia email a ogni utente registrato con il report dei SOLI profili assegnati.
- Admin (fabio) riceve il report di TUTTI i profili
- Utenti normali ricevono solo i profili con created_by = loro username + hr-roma (condiviso)

Uso:
    python send_reports.py

Variabili d'ambiente richieste:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
"""

import json
import logging
import os
import smtplib
import ssl
import glob
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent

log = logging.getLogger("email_report")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [EMAIL] %(message)s", datefmt="%H:%M:%S")


def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_user_profiles(username, role, profiles):
    """Restituisce i profili visibili per un utente."""
    visible = {}
    for pid, p in profiles.items():
        if role == "admin":
            visible[pid] = p
        elif pid == "hr-roma":
            visible[pid] = p  # condiviso
        elif p.get("created_by") == username:
            visible[pid] = p
    return visible


def build_report_html(user_profiles):
    """Genera il corpo email HTML con i risultati per i profili dell'utente."""
    today = datetime.now().strftime("%d/%m/%Y")
    
    html = f"""
    <html>
    <head><style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #333; max-width: 700px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #6c5ce7; font-size: 22px; }}
        h2 {{ color: #2d3436; font-size: 16px; margin-top: 24px; border-bottom: 2px solid #6c5ce7; padding-bottom: 6px; }}
        .stats {{ display: flex; gap: 16px; margin: 12px 0; }}
        .stat {{ background: #f5f6fa; padding: 10px 16px; border-radius: 8px; text-align: center; }}
        .stat .num {{ font-size: 20px; font-weight: 700; color: #6c5ce7; }}
        .stat .label {{ font-size: 11px; color: #636e72; }}
        table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
        th {{ background: #6c5ce7; color: #fff; padding: 8px 10px; text-align: left; font-size: 11px; text-transform: uppercase; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #dfe6e9; }}
        tr:hover td {{ background: #f5f6fa; }}
        a {{ color: #6c5ce7; text-decoration: none; }}
        .footer {{ margin-top: 30px; padding-top: 16px; border-top: 1px solid #dfe6e9; font-size: 12px; color: #636e72; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
        .badge-new {{ background: #00b894; color: #fff; }}
        .badge-indeed {{ background: #2e5aac; color: #fff; }}
        .badge-linkedin {{ background: #0077b5; color: #fff; }}
        .badge-google {{ background: #ea4335; color: #fff; }}
    </style></head>
    <body>
    <h1>📋 TA Monitor — Report {today}</h1>
    """

    total_new = 0
    
    for pid, profile in user_profiles.items():
        name = profile.get("name", pid)
        location = profile.get("location", "")
        
        # Load summary
        summary = None
        for sp in [BASE_DIR / "output" / pid / "summary.json", BASE_DIR / "output" / "summary.json"]:
            if sp.exists():
                try:
                    summary = load_json(sp)
                    break
                except Exception:
                    pass
        
        # Load latest jobs
        jobs = []
        job_dir = BASE_DIR / "output" / pid
        if not job_dir.exists():
            job_dir = BASE_DIR / "output"
        json_files = sorted(job_dir.glob("ta_jobs_*.json"), reverse=True)
        if json_files:
            try:
                with open(json_files[0], "r", encoding="utf-8") as f:
                    jobs = json.load(f)
            except Exception:
                pass

        html += f'<h2>🔬 {name} <span style="font-size:12px;color:#636e72;font-weight:normal">📍 {location}</span></h2>'
        
        if summary:
            new_jobs = summary.get("new_jobs", 0)
            total_new += new_jobs
            html += f"""
            <div class="stats">
                <div class="stat"><div class="num">{summary.get('total_raw', 0)}</div><div class="label">Grezzo</div></div>
                <div class="stat"><div class="num" style="color:#e74c3c">{summary.get('filtered_irrelevant', 0)}</div><div class="label">Scartati</div></div>
                <div class="stat"><div class="num" style="color:#00b894">{new_jobs}</div><div class="label">Nuove</div></div>
                <div class="stat"><div class="num" style="color:#636e72">{summary.get('duplicates_removed', 0)}</div><div class="label">Duplicati</div></div>
            </div>
            """
        
        if jobs:
            html += """
            <table>
            <thead><tr><th>Titolo</th><th>Azienda</th><th>Canale</th><th>Luogo</th><th>Data</th><th>Link</th></tr></thead>
            <tbody>
            """
            for j in jobs[:20]:  # Max 20 per email
                title = j.get("title", "")
                company = j.get("company", "")
                site = (j.get("site", "") or "").lower()
                loc = j.get("location") or j.get("city") or ""
                date_p = ""
                raw = j.get("date_posted")
                if raw and str(raw) not in ("None", "NaT", "nan", ""):
                    date_p = str(raw)[:10]
                
                url = ""
                for k in ["job_url", "job_url_direct", "url"]:
                    if j.get(k) and str(j[k]).startswith("http"):
                        url = j[k]
                        break
                
                badge_cls = "badge-linkedin" if "linkedin" in site else "badge-indeed" if "indeed" in site else "badge-google" if "google" in site else ""
                
                html += f"""<tr>
                    <td><strong>{title}</strong></td>
                    <td>{company}</td>
                    <td><span class="badge {badge_cls}">{site}</span></td>
                    <td style="font-size:12px">{loc}</td>
                    <td style="font-size:12px;white-space:nowrap">{date_p}</td>
                    <td>{'<a href="' + url + '" target="_blank">↗ Apri</a>' if url else '—'}</td>
                </tr>"""
            
            html += "</tbody></table>"
            if len(jobs) > 20:
                html += f'<p style="font-size:12px;color:#636e72">...e altri {len(jobs) - 20} annunci. Vedi tutti sulla dashboard.</p>'
        else:
            html += '<p style="color:#636e72;font-size:13px">Nessun risultato per questo profilo.</p>'

    html += f"""
    <div class="footer">
        <p>📊 <a href="https://fabiodeluca1977.github.io/ta-monitor/analytics.html">Vedi Analytics</a> | 
           🔧 <a href="https://fabiodeluca1977.github.io/ta-monitor/">Apri Dashboard</a></p>
        <p>Report generato automaticamente da TA Monitor — {today}</p>
    </div>
    </body></html>
    """
    
    return html, total_new


def send_email(smtp_host, smtp_port, smtp_user, smtp_password, to_email, subject, html_body, attachments=None):
    """Invia email via SMTP."""
    msg = MIMEMultipart("alternative")
    msg["From"] = f"TA Monitor <{smtp_user}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    
    # Plain text fallback
    msg.attach(MIMEText("Apri questa email in un client che supporta HTML.", "plain"))
    msg.attach(MIMEText(html_body, "html"))
    
    # Attachments
    if attachments:
        for fpath in attachments:
            if Path(fpath).exists():
                with open(fpath, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={Path(fpath).name}")
                msg.attach(part)
    
    context = ssl.create_default_context()
    try:
        port = int(smtp_port)
        if port == 465:
            with smtplib.SMTP_SSL(smtp_host, port, context=context) as server:
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, to_email, msg.as_string())
        else:
            with smtplib.SMTP(smtp_host, port) as server:
                server.starttls(context=context)
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, to_email, msg.as_string())
        return True
    except Exception as e:
        log.error(f"  Errore invio a {to_email}: {e}")
        return False


def main():
    # SMTP config from environment
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = os.environ.get("SMTP_PORT", "465")
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    
    if not all([smtp_host, smtp_user, smtp_password]):
        log.warning("SMTP non configurato, skip email")
        return
    
    # Load data
    users_data = load_json(BASE_DIR / "users.json")
    profiles_data = load_json(BASE_DIR / "profiles.json")
    
    if not users_data or not profiles_data:
        log.error("users.json o profiles.json non trovati")
        return
    
    profiles = profiles_data.get("profiles", {})
    users = users_data.get("users", [])
    
    today = datetime.now().strftime("%d/%m/%Y")
    
    for user in users:
        email = user.get("email", "")
        username = user.get("username", "")
        role = user.get("role", "user")
        
        if not email:
            log.info(f"  {username}: nessuna email, skip")
            continue
        
        # Get user's visible profiles
        user_profiles = get_user_profiles(username, role, profiles)
        
        if not user_profiles:
            log.info(f"  {username}: nessun profilo, skip")
            continue
        
        log.info(f"▸ {username} ({email}) — {len(user_profiles)} profili")
        
        # Build report
        html_body, total_new = build_report_html(user_profiles)
        
        # Find Excel attachments for user's profiles
        attachments = []
        for pid in user_profiles:
            xlsx_files = glob.glob(f"output/{pid}/*.xlsx") + glob.glob("output/*.xlsx")
            attachments.extend(xlsx_files)
        attachments = list(set(attachments))[:5]  # Max 5 attachments
        
        # Send
        subject = f"📋 TA Monitor — Report {today}"
        if total_new > 0:
            subject += f" ({total_new} nuove posizioni)"
        
        ok = send_email(smtp_host, smtp_port, smtp_user, smtp_password, email, subject, html_body, attachments)
        if ok:
            log.info(f"  ✓ Email inviata a {email}")
        else:
            log.error(f"  ✗ Errore invio a {email}")
    
    log.info("✓ Email report completato")


if __name__ == "__main__":
    main()
