#!/usr/bin/env python3
"""Migliora i profili MSL e Regulatory con keyword aggiuntive dal contesto reale."""
import json

with open("profiles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# === KEYWORD AGGIUNTIVE PER MSL ===
msl_extra_search = [
    "medical advisor",
    "medical advisor oncology",
    "medical advisor solid tumors",
    "translational research",
    "scientific affairs",
    "medical communication",
    "medical writing",
    "KOL engagement",
    "KOL management",
    "scientific exchange",
    "evidence generation",
    "real world evidence",
    "RWE",
    "clinical development",
    "medical device specialist",
    "ATMP",
    "advanced therapies"
]

msl_extra_patterns = [
    {"pattern": "medical\\s+advisor.*(?:oncol|tumor|solid)", "category": "Medical Advisor Oncology", "note": ""},
    {"pattern": "medical\\s+advisor", "category": "Medical Advisor", "note": ""},
    {"pattern": "translational\\s+(?:research|oncol)", "category": "Translational Research", "note": ""},
    {"pattern": "KOL\\s+(?:engagement|management|liaison)", "category": "KOL", "note": ""},
    {"pattern": "scientific\\s+(?:exchange|communication|affairs)", "category": "Scientific Affairs", "note": ""},
    {"pattern": "evidence\\s+generation", "category": "Evidence Generation", "note": ""},
    {"pattern": "real\\s+world\\s+evidence|\\bRWE\\b", "category": "RWE", "note": ""},
    {"pattern": "clinical\\s+development", "category": "Clinical Development", "note": ""},
    {"pattern": "medical\\s+(?:communication|writing)", "category": "Medical Communication", "note": ""},
    {"pattern": "(?:mRNA|advanced\\s+therap|ATMP)", "category": "Advanced Therapies", "note": ""},
    {"pattern": "(?:oncol|tumor|solid\\s+tumor|melanom|immuno.oncol)", "category": "Oncology", "note": ""},
    {"pattern": "chief\\s+scientific|CSO", "category": "Scientific Leadership", "note": ""},
    {"pattern": "preclinical.*(?:oncol|research|develop)", "category": "Preclinical", "note": ""},
    {"pattern": "medical\\s+device.*(?:specialist|manager|regulatory)", "category": "Medical Device", "note": ""}
]

msl_extra_broad = [
    "medical advisor", "medical advisor oncology", "solid tumors",
    "translational research", "KOL engagement", "KOL management",
    "scientific exchange", "evidence generation", "real world evidence",
    "RWE", "clinical development", "mRNA therapy", "ATMP",
    "advanced therapies", "immuno-oncology", "melanoma",
    "preclinical oncology", "medical device", "CSO",
    "scientific leadership", "cross-functional"
]

# === KEYWORD AGGIUNTIVE PER REGULATORY ===
reg_extra_search = [
    "regulatory affairs oncology",
    "regulatory strategy",
    "CTD dossier",
    "MAA submission",
    "EMA submission",
    "AIFA compliance",
    "clinical trial authorization",
    "CTA submission",
    "regulatory medical device",
    "notified body",
    "CE marking",
    "MDR compliance",
    "post-market surveillance",
    "scientific advice EMA",
    "variation management"
]

reg_extra_patterns = [
    {"pattern": "regulatory.*(?:oncol|pharma|biotech)", "category": "Regulatory Pharma", "note": ""},
    {"pattern": "(?:MAA|CTA|CTD)\\s+(?:submission|dossier|preparation)", "category": "Regulatory Submission", "note": ""},
    {"pattern": "(?:EMA|AIFA|FDA).*(?:submission|approval|compliance|advice)", "category": "Health Authority", "note": ""},
    {"pattern": "CE\\s+mark|MDR\\s+compliance|notified\\s+body", "category": "Medical Device Regulatory", "note": ""},
    {"pattern": "post.market\\s+surveillance", "category": "Post-Market", "note": ""},
    {"pattern": "variation\\s+management", "category": "Regulatory Operations", "note": ""},
    {"pattern": "clinical\\s+trial\\s+authoriz", "category": "CTA", "note": ""},
    {"pattern": "scientific\\s+advice", "category": "Regulatory Strategy", "note": ""},
    {"pattern": "regulatory\\s+strateg", "category": "Regulatory Strategy", "note": ""}
]

reg_extra_broad = [
    "regulatory oncology", "CTD", "MAA", "CTA",
    "EMA submission", "AIFA", "CE marking", "MDR",
    "notified body", "post-market surveillance",
    "variation management", "scientific advice",
    "regulatory strategy", "clinical trial authorization"
]

def add_unique_strings(existing, new_items):
    """Aggiunge stringhe senza duplicati (case-insensitive)."""
    existing_lower = {s.lower() for s in existing}
    added = 0
    for item in new_items:
        if item.lower() not in existing_lower:
            existing.append(item)
            existing_lower.add(item.lower())
            added += 1
    return added

def add_unique_patterns(existing, new_patterns):
    """Aggiunge pattern senza duplicati."""
    existing_pats = {p["pattern"] for p in existing}
    added = 0
    for p in new_patterns:
        if p["pattern"] not in existing_pats:
            existing.append(p)
            existing_pats.add(p["pattern"])
            added += 1
    return added

# Aggiorna profili MSL
msl_ids = [k for k in data["profiles"] if k.startswith("msl-")]
for pid in msl_ids:
    f = data["profiles"][pid]["filters"]
    s1 = add_unique_strings(f["search_terms"], msl_extra_search)
    p1 = add_unique_patterns(f["relevant_patterns"], msl_extra_patterns)
    b1 = add_unique_strings(f["broad_keywords"], msl_extra_broad)
    print(f"  {pid}: +{s1} search, +{p1} pattern, +{b1} broad")

# Aggiorna profili Regulatory
reg_ids = [k for k in data["profiles"] if k.startswith("reg-")]
for pid in reg_ids:
    f = data["profiles"][pid]["filters"]
    s1 = add_unique_strings(f["search_terms"], reg_extra_search)
    p1 = add_unique_patterns(f["relevant_patterns"], reg_extra_patterns)
    b1 = add_unique_strings(f["broad_keywords"], reg_extra_broad)
    print(f"  {pid}: +{s1} search, +{p1} pattern, +{b1} broad")

with open("profiles.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✓ Aggiornati {len(msl_ids)} profili MSL e {len(reg_ids)} profili Regulatory")
print(f"Profili totali: {list(data['profiles'].keys())}")
