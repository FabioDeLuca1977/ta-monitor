#!/usr/bin/env python3
"""Rifocalizza profili MSL: rimuove ISF, rafforza Medical Advisor/MSL."""
import json

with open("profiles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Keyword da RIMUOVERE (ISF = ruolo commerciale, non pertinente)
remove_search = {
    "informatore scientifico",
    "informatore medico scientifico",
    "ISF",
}

remove_patterns_containing = {
    "informator",
    "\\bISF\\b",
}

# Keyword da AGGIUNGERE (Medical Advisor / MSL senior)
add_search = [
    "medical advisor oncology",
    "medical advisor solid tumors",
    "medical advisor hematology",
    "medical advisor immunology",
    "senior medical advisor",
    "associate medical director",
    "medical affairs manager",
    "medical affairs director",
    "head of medical affairs",
    "scientific director",
    "chief medical officer",
    "CMO pharma",
    "medical lead",
    "therapeutic area lead",
    "medical science liaison oncology",
    "MSL oncology",
    "MSL manager",
    "field medical director",
    "regional medical liaison",
    "medical affairs specialist",
    "clinical science liaison",
    "scientific affairs manager",
    "real world evidence manager",
    "HEOR manager",
    "market access manager pharma",
]

add_patterns = [
    {"pattern": "senior\\s+medical\\s+advisor", "category": "Medical Advisor Senior", "note": ""},
    {"pattern": "associate\\s+medical\\s+director", "category": "Medical Director", "note": ""},
    {"pattern": "medical\\s+affairs\\s+(?:manager|director|head|lead)", "category": "Medical Affairs Leadership", "note": ""},
    {"pattern": "head\\s+(?:of\\s+)?medical\\s+affairs", "category": "Medical Affairs Leadership", "note": ""},
    {"pattern": "scientific\\s+director", "category": "Scientific Leadership", "note": ""},
    {"pattern": "chief\\s+medical\\s+officer|\\bCMO\\b", "category": "C-Level Medical", "note": ""},
    {"pattern": "medical\\s+lead", "category": "Medical Lead", "note": ""},
    {"pattern": "therapeutic\\s+area\\s+(?:lead|manager|director)", "category": "TA Lead", "note": ""},
    {"pattern": "MSL.*(?:oncol|manager|director|lead)", "category": "MSL Senior", "note": ""},
    {"pattern": "field\\s+medical\\s+(?:director|lead|manager)", "category": "Field Medical Leadership", "note": ""},
    {"pattern": "regional\\s+medical\\s+(?:liaison|manager|lead)", "category": "Regional Medical", "note": ""},
    {"pattern": "clinical\\s+science\\s+liaison", "category": "Clinical Science", "note": ""},
    {"pattern": "scientific\\s+affairs\\s+(?:manager|director)", "category": "Scientific Affairs", "note": ""},
    {"pattern": "real\\s+world\\s+evidence\\s+(?:manager|director|lead)", "category": "RWE", "note": ""},
    {"pattern": "HEOR\\s+(?:manager|director|specialist)", "category": "HEOR", "note": ""},
    {"pattern": "market\\s+access\\s+(?:manager|director).*(?:pharma|medical)", "category": "Market Access", "note": ""},
    {"pattern": "medical\\s+advisor.*(?:biotech|pharma|biopharma)", "category": "Medical Advisor", "note": ""},
]

add_broad = [
    "senior medical advisor", "associate medical director",
    "head of medical affairs", "scientific director",
    "chief medical officer", "CMO", "medical lead",
    "therapeutic area lead", "MSL oncology", "MSL manager",
    "field medical director", "clinical science liaison",
    "scientific affairs manager", "RWE manager", "HEOR manager",
    "market access pharma", "medical affairs leadership",
    "biopharma", "biotech medical"
]

# Blacklist ISF (è un ruolo commerciale)
add_blacklist = [
    {"pattern": "informator.*(?:scientifico|medico|farmac)", "category": "ISF (commerciale)"},
    {"pattern": "\\bISF\\b", "category": "ISF (commerciale)"},
    {"pattern": "informazione\\s+scientifica", "category": "ISF (commerciale)"},
    {"pattern": "propaganda\\s+(?:medica|scientifica)", "category": "ISF (commerciale)"},
]

for pid, profile in data["profiles"].items():
    if not pid.startswith("msl-"):
        continue
    
    f = profile["filters"]
    
    # 1. Rimuovi search terms ISF
    original_len = len(f["search_terms"])
    f["search_terms"] = [s for s in f["search_terms"] if s not in remove_search]
    removed_search = original_len - len(f["search_terms"])
    
    # 2. Rimuovi pattern ISF
    original_len = len(f["relevant_patterns"])
    f["relevant_patterns"] = [p for p in f["relevant_patterns"] 
                               if not any(rem in p["pattern"] for rem in remove_patterns_containing)]
    removed_patterns = original_len - len(f["relevant_patterns"])
    
    # 3. Rimuovi broad ISF
    f["broad_keywords"] = [b for b in f["broad_keywords"] 
                           if b.lower() not in {"informatore scientifico", "isf"}]
    
    # 4. Aggiungi nuove search terms
    existing_lower = {s.lower() for s in f["search_terms"]}
    added_s = 0
    for s in add_search:
        if s.lower() not in existing_lower:
            f["search_terms"].append(s)
            existing_lower.add(s.lower())
            added_s += 1
    
    # 5. Aggiungi nuovi pattern
    existing_pats = {p["pattern"] for p in f["relevant_patterns"]}
    added_p = 0
    for p in add_patterns:
        if p["pattern"] not in existing_pats:
            f["relevant_patterns"].append(p)
            existing_pats.add(p["pattern"])
            added_p += 1
    
    # 6. Aggiungi nuovi broad
    existing_broad = {b.lower() for b in f["broad_keywords"]}
    added_b = 0
    for b in add_broad:
        if b.lower() not in existing_broad:
            f["broad_keywords"].append(b)
            existing_broad.add(b.lower())
            added_b += 1
    
    # 7. Aggiungi ISF alla blacklist
    existing_bl = {p["pattern"] for p in f["blacklist"]}
    added_bl = 0
    for bl in add_blacklist:
        if bl["pattern"] not in existing_bl:
            f["blacklist"].append(bl)
            existing_bl.add(bl["pattern"])
            added_bl += 1
    
    print(f"  {pid}:")
    print(f"    Rimossi: {removed_search} search ISF, {removed_patterns} pattern ISF")
    print(f"    Aggiunti: +{added_s} search, +{added_p} pattern, +{added_b} broad, +{added_bl} blacklist")
    print(f"    Totale: {len(f['search_terms'])} search, {len(f['relevant_patterns'])} pattern, {len(f['broad_keywords'])} broad, {len(f['blacklist'])} blacklist")

with open("profiles.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✓ Profili MSL rifocalizzati su Medical Advisor / MSL senior")
