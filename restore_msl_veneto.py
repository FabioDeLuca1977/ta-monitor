#!/usr/bin/env python3
"""Ripristina il profilo msl-veneto alla versione originale."""
import json

with open("profiles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

data["profiles"]["msl-veneto"] = {
    "name": "Medical Liaison — Veneto",
    "description": "Monitoraggio posizioni Medical Science Liaison, Medical Affairs e ruoli scientifici pharma in Veneto",
    "auto_scan": True,
    "location": "Veneto, Italia",
    "hours_old": 168,
    "created": "2026-03-07T00:00:00",
    "created_by": "gabriellabellavia",
    "filters": {
        "search_terms": [
            "medical science liaison","medical liaison","MSL",
            "medical affairs","medical advisor","medical manager",
            "field medical advisor","regional medical advisor",
            "scientific liaison","scientific advisor","clinical liaison",
            "medical information","pharmacovigilance","farmacovigilanza",
            "informatore scientifico","informatore medico scientifico","ISF",
            "drug safety","medical director","clinical research associate","CRA monitor"
        ],
        "relevant_patterns": [
            {"pattern":"medical\\s*(?:science)?\\s*liaison","category":"MSL","note":""},
            {"pattern":"\\bMSL\\b","category":"MSL","note":"Acronimo"},
            {"pattern":"medical\\s+affair","category":"Medical Affairs","note":""},
            {"pattern":"medical\\s+advisor","category":"Medical Affairs","note":""},
            {"pattern":"medical\\s+manager","category":"Medical Affairs","note":""},
            {"pattern":"medical\\s+director","category":"Medical Affairs","note":""},
            {"pattern":"field\\s+medical","category":"MSL","note":""},
            {"pattern":"regional\\s+medical\\s+advis","category":"MSL","note":""},
            {"pattern":"scientific\\s+(?:liaison|advisor|manager)","category":"Scientific Affairs","note":""},
            {"pattern":"clinical\\s+liaison","category":"Clinical","note":""},
            {"pattern":"clinical\\s+research\\s+associate","category":"Clinical","note":""},
            {"pattern":"\\bCRA\\b.*(?:monitor|clinical)","category":"Clinical","note":""},
            {"pattern":"pharmacovigilan","category":"Farmacovigilanza","note":""},
            {"pattern":"farmacovigilanz","category":"Farmacovigilanza","note":""},
            {"pattern":"drug\\s+safety","category":"Farmacovigilanza","note":""},
            {"pattern":"informator.*(?:scientifico|medico)","category":"ISF","note":""},
            {"pattern":"\\bISF\\b","category":"ISF","note":"Acronimo"},
            {"pattern":"medical\\s+information","category":"Medical Information","note":""},
            {"pattern":"(?:medical|scientific)\\s+(?:communication|writing)","category":"Medical Communication","note":""},
            {"pattern":"regulatory\\s+affair","category":"Regulatory","note":""},
            {"pattern":"(?:clinical|medical)\\s+(?:monitor|specialist|associate)","category":"Clinical","note":""},
            {"pattern":"HEOR|health\\s+econom","category":"HEOR","note":""},
            {"pattern":"market\\s+access.*(?:pharma|medical|health)","category":"Market Access","note":""}
        ],
        "broad_keywords": [
            "medical liaison","MSL","medical affairs","medical advisor",
            "medical science liaison","field medical","scientific liaison",
            "pharmacovigilance","farmacovigilanza","drug safety",
            "informatore scientifico","ISF","clinical research",
            "CRA","medical information","regulatory affairs",
            "medical director","medical manager","clinical liaison",
            "HEOR","market access","medical communication",
            "scientific advisor","medical writing"
        ],
        "blacklist": [
            {"pattern":"sviluppat","category":"Tech / IT"},
            {"pattern":"developer","category":"Tech / IT"},
            {"pattern":"software\\s+engineer","category":"Tech / IT"},
            {"pattern":"full\\s*stack","category":"Tech / IT"},
            {"pattern":"front\\s*end","category":"Tech / IT"},
            {"pattern":"back\\s*end","category":"Tech / IT"},
            {"pattern":"data\\s+(?:engineer|scientist|analyst)","category":"Tech / IT"},
            {"pattern":"devops","category":"Tech / IT"},
            {"pattern":"cloud.*(?:architect|engineer)","category":"Tech / IT"},
            {"pattern":"camerier","category":"Food / Retail"},
            {"pattern":"barista","category":"Food / Retail"},
            {"pattern":"cuoc[oa]","category":"Food / Retail"},
            {"pattern":"commess[oa]","category":"Food / Retail"},
            {"pattern":"cassier","category":"Food / Retail"},
            {"pattern":"store\\s+manager","category":"Food / Retail"},
            {"pattern":"ristorazion","category":"Food / Retail"},
            {"pattern":"magazzin","category":"Operations"},
            {"pattern":"operai[oa]","category":"Operations"},
            {"pattern":"muratore","category":"Operations"},
            {"pattern":"idraulic","category":"Operations"},
            {"pattern":"elettricist","category":"Operations"},
            {"pattern":"autista","category":"Operations"},
            {"pattern":"manutenz","category":"Operations"},
            {"pattern":"contabil","category":"Finance"},
            {"pattern":"accountant","category":"Finance"},
            {"pattern":"agente.*(?:commerc|immobili|assicur)","category":"Sales"},
            {"pattern":"graphic\\s*design","category":"Marketing"},
            {"pattern":"social\\s+media","category":"Marketing"},
            {"pattern":"copywriter","category":"Marketing"},
            {"pattern":"seo\\b","category":"Marketing"},
            {"pattern":"segretari[oa]","category":"Admin"},
            {"pattern":"receptionist","category":"Admin"},
            {"pattern":"hr\\s+(?:manager|generalist|specialist|business)","category":"HR (non pertinente)"},
            {"pattern":"talent\\s+acquisition","category":"HR (non pertinente)"},
            {"pattern":"recruiter|recruiting","category":"HR (non pertinente)"},
            {"pattern":"selezione.*personale","category":"HR (non pertinente)"},
            {"pattern":"concorso\\s+pubblic","category":"Bandi pubblici"},
            {"pattern":"bando","category":"Bandi pubblici"},
            {"pattern":"selezione\\s+pubblica","category":"Bandi pubblici"}
        ],
        "agencies": [
            "Adami & Associati","Adecco","Ali Lavoro","Altaïde","Alti Profili",
            "Amrop","Antal","Articolo1","Badenoch + Clark","Boyden",
            "Carter & Benson","Catenon","e-work","Egon Zehnder","Etjca",
            "Experis","Gi Group","Grafton","Hays","Heidrick & Struggles",
            "Humangest","Hunters Group","In Job","IQM Selezione","Iqvia",
            "Kelly Services","Key2People","Korn Ferry","Lavoropiù","MAW",
            "Manpower","MF Consultant","Michael Page","Odgers Berndtson",
            "Openjobmetis","Orienta","Page Personnel","Parexel","PharmaLex",
            "PharmaPoint","Quanta","Randstad","Robert Half","Robert Walters",
            "Russell Reynolds","S&you","Spencer Stuart","Spring Professional",
            "Stryker","Synergie","Temporary","Umana","W Executive"
        ]
    }
}

with open("profiles.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✓ Profilo msl-veneto ripristinato alla versione originale")
print(f"  - 21 keyword")
print(f"  - 23 pattern rilevanza")
print(f"  - 24 broad keywords")
print(f"  - 39 blacklist")
print(f"  - 53 agenzie (originali + nuove)")
