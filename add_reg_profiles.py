#!/usr/bin/env python3
"""Aggiunge 4 profili Regulatory Affairs al profiles.json."""
import json

with open("profiles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

reg_filters = {
    "search_terms": [
        "regulatory affairs",
        "regulatory manager",
        "regulatory specialist",
        "regulatory affairs manager",
        "regulatory affairs specialist",
        "regulatory affairs director",
        "responsabile affari regolatori",
        "specialista affari regolatori",
        "quality assurance pharma",
        "QA manager pharma",
        "qualified person",
        "persona qualificata",
        "GMP compliance",
        "GxP specialist",
        "drug registration",
        "registrazione farmaci",
        "CMC regulatory",
        "regulatory submission",
        "clinical regulatory",
        "regulatory compliance",
        "labeling specialist",
        "medical device regulatory"
    ],
    "relevant_patterns": [
        {"pattern": "regulatory\\s+affair", "category": "Regulatory Affairs", "note": ""},
        {"pattern": "regulatory\\s+manager", "category": "Regulatory Affairs", "note": ""},
        {"pattern": "regulatory\\s+specialist", "category": "Regulatory Affairs", "note": ""},
        {"pattern": "regulatory\\s+director", "category": "Regulatory Affairs", "note": ""},
        {"pattern": "regulatory\\s+officer", "category": "Regulatory Affairs", "note": ""},
        {"pattern": "regulatory\\s+(?:compliance|submission|strategy)", "category": "Regulatory Affairs", "note": ""},
        {"pattern": "affari\\s+regolator", "category": "Regulatory Affairs", "note": ""},
        {"pattern": "responsabile\\s+regolator", "category": "Regulatory Affairs", "note": ""},
        {"pattern": "qualified\\s+person", "category": "Quality / QP", "note": ""},
        {"pattern": "persona\\s+qualificata", "category": "Quality / QP", "note": ""},
        {"pattern": "quality\\s+assurance.*(?:pharma|farmac|GMP)", "category": "Quality Assurance", "note": ""},
        {"pattern": "QA\\s+(?:manager|specialist|director).*(?:pharma|farmac|GMP)", "category": "Quality Assurance", "note": ""},
        {"pattern": "GMP\\s+(?:compliance|specialist|manager)", "category": "GMP", "note": ""},
        {"pattern": "GxP\\s+(?:specialist|compliance|manager)", "category": "GMP", "note": ""},
        {"pattern": "\\bCMC\\b.*regulatory", "category": "CMC", "note": ""},
        {"pattern": "drug\\s+registration", "category": "Drug Registration", "note": ""},
        {"pattern": "registrazione\\s+farmac", "category": "Drug Registration", "note": ""},
        {"pattern": "clinical\\s+regulatory", "category": "Clinical Regulatory", "note": ""},
        {"pattern": "labeling\\s+(?:specialist|manager)", "category": "Labeling", "note": ""},
        {"pattern": "medical\\s+device.*regulatory", "category": "Medical Device", "note": ""},
        {"pattern": "dispositiv.*medic.*regolator", "category": "Medical Device", "note": ""},
        {"pattern": "(?:EMA|AIFA|FDA)\\s+(?:submission|compliance|specialist)", "category": "Regulatory Affairs", "note": ""},
        {"pattern": "dossier\\s+(?:registr|regulat|CTD)", "category": "Drug Registration", "note": ""},
        {"pattern": "pharmacovigilan", "category": "Farmacovigilanza", "note": ""},
        {"pattern": "farmacovigilanz", "category": "Farmacovigilanza", "note": ""}
    ],
    "broad_keywords": [
        "regulatory affairs", "regulatory manager", "regulatory specialist",
        "affari regolatori", "quality assurance", "QA pharma",
        "qualified person", "persona qualificata", "GMP", "GxP",
        "drug registration", "registrazione farmaci", "CMC",
        "regulatory compliance", "regulatory submission",
        "clinical regulatory", "labeling", "medical device",
        "EMA", "AIFA", "CTD", "dossier", "pharmacovigilance",
        "farmacovigilanza", "regulatory strategy"
    ],
    "blacklist": [
        {"pattern": "sviluppat", "category": "Tech / IT"},
        {"pattern": "developer", "category": "Tech / IT"},
        {"pattern": "software\\s+engineer", "category": "Tech / IT"},
        {"pattern": "full\\s*stack", "category": "Tech / IT"},
        {"pattern": "front\\s*end", "category": "Tech / IT"},
        {"pattern": "back\\s*end", "category": "Tech / IT"},
        {"pattern": "data\\s+(?:engineer|scientist|analyst)", "category": "Tech / IT"},
        {"pattern": "devops", "category": "Tech / IT"},
        {"pattern": "cloud.*(?:architect|engineer)", "category": "Tech / IT"},
        {"pattern": "camerier", "category": "Food / Retail"},
        {"pattern": "barista", "category": "Food / Retail"},
        {"pattern": "cuoc[oa]", "category": "Food / Retail"},
        {"pattern": "commess[oa]", "category": "Food / Retail"},
        {"pattern": "cassier", "category": "Food / Retail"},
        {"pattern": "store\\s+manager", "category": "Food / Retail"},
        {"pattern": "ristorazion", "category": "Food / Retail"},
        {"pattern": "magazzin", "category": "Operations"},
        {"pattern": "operai[oa]", "category": "Operations"},
        {"pattern": "muratore", "category": "Operations"},
        {"pattern": "idraulic", "category": "Operations"},
        {"pattern": "elettricist", "category": "Operations"},
        {"pattern": "autista", "category": "Operations"},
        {"pattern": "manutenz", "category": "Operations"},
        {"pattern": "contabil", "category": "Finance"},
        {"pattern": "accountant", "category": "Finance"},
        {"pattern": "agente.*(?:commerc|immobili|assicur)", "category": "Sales"},
        {"pattern": "graphic\\s*design", "category": "Marketing"},
        {"pattern": "social\\s+media", "category": "Marketing"},
        {"pattern": "copywriter", "category": "Marketing"},
        {"pattern": "seo\\b", "category": "Marketing"},
        {"pattern": "segretari[oa]", "category": "Admin"},
        {"pattern": "receptionist", "category": "Admin"},
        {"pattern": "hr\\s+(?:manager|generalist|specialist|business)", "category": "HR (non pertinente)"},
        {"pattern": "talent\\s+acquisition", "category": "HR (non pertinente)"},
        {"pattern": "recruiter|recruiting", "category": "HR (non pertinente)"},
        {"pattern": "selezione.*personale", "category": "HR (non pertinente)"},
        {"pattern": "concorso\\s+pubblic", "category": "Bandi pubblici"},
        {"pattern": "bando", "category": "Bandi pubblici"},
        {"pattern": "selezione\\s+pubblica", "category": "Bandi pubblici"}
    ],
    "agencies": [
        "Randstad", "Adecco", "Manpower", "Gi Group",
        "Hays", "Page Group", "Michael Page",
        "Kelly", "Hunters", "Spring Professional",
        "PharmaLex", "Iqvia", "Parexel"
    ]
}

regions = {
    "reg-veneto":  ("Regulatory Affairs — Veneto",  "Veneto, Italia", "Veneto"),
    "reg-lombardia": ("Regulatory Affairs — Lombardia", "Milano, Lombardia, Italia", "Lombardia"),
    "reg-emilia":  ("Regulatory Affairs — Emilia-Romagna", "Bologna, Emilia-Romagna, Italia", "Emilia-Romagna"),
    "reg-lazio":   ("Regulatory Affairs — Lazio", "Roma, Lazio, Italia", "Lazio"),
}

added = []
for pid, (name, location, region) in regions.items():
    if pid in data["profiles"]:
        print(f"  SKIP: {pid} esiste già")
        continue
    profile = {
        "name": name,
        "description": f"Monitoraggio posizioni Regulatory Affairs, QA, GMP e registrazione farmaci in {region}",
        "auto_scan": True,
        "location": location,
        "hours_old": 168,
        "created": "2026-03-07T00:00:00",
        "created_by": "fabio",
        "filters": json.loads(json.dumps(reg_filters))
    }
    data["profiles"][pid] = profile
    added.append(pid)

with open("profiles.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Aggiunti {len(added)} profili: {', '.join(added)}")
print(f"Profili totali: {list(data['profiles'].keys())}")
