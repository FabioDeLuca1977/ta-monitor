#!/usr/bin/env python3
"""Aggiunge blacklist ai profili MSL per rimuovere falsi positivi."""
import json

with open("profiles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

extra_blacklist = [
    {"pattern": "sommelier", "category": "Food / Retail"},
    {"pattern": "victim\\s+advocate", "category": "Non pertinente"},
    {"pattern": "marketing\\s+expert", "category": "Marketing"},
    {"pattern": "marketing\\s+manager", "category": "Marketing"},
    {"pattern": "laboratory\\s+technician", "category": "Lab Tecnico"},
    {"pattern": "lab\\s+technician", "category": "Lab Tecnico"},
    {"pattern": "waiter|waitress", "category": "Food / Retail"},
    {"pattern": "chef\\b", "category": "Food / Retail"},
    {"pattern": "kitchen", "category": "Food / Retail"},
    {"pattern": "housekeep", "category": "Hospitality"},
    {"pattern": "concierge", "category": "Hospitality"},
    {"pattern": "front\\s+desk", "category": "Hospitality"},
    {"pattern": "hotel\\s+manager", "category": "Hospitality"},
    {"pattern": "sales\\s+(?:representative|manager|director|executive)", "category": "Sales"},
    {"pattern": "account\\s+manager", "category": "Sales"},
    {"pattern": "business\\s+develop", "category": "Sales"},
    {"pattern": "area\\s+manager", "category": "Sales"},
    {"pattern": "informatore.*commercial", "category": "Sales"},
    {"pattern": "tecnico.*laboratorio", "category": "Lab Tecnico"},
    {"pattern": "analista.*laboratorio", "category": "Lab Tecnico"},
    {"pattern": "biologo.*laboratorio", "category": "Lab Tecnico"},
    {"pattern": "quality\\s+control(?!.*pharma)", "category": "QC generico"},
    {"pattern": "produzione|production\\s+(?:operator|manager)", "category": "Production"},
    {"pattern": "supply\\s+chain", "category": "Operations"},
    {"pattern": "logistic", "category": "Operations"},
    {"pattern": "warehouse", "category": "Operations"},
    {"pattern": "infermier", "category": "Nursing"},
    {"pattern": "nurse\\b", "category": "Nursing"},
    {"pattern": "OSS\\b", "category": "Nursing"},
    {"pattern": "fisioterapi", "category": "Healthcare"},
    {"pattern": "veterinar", "category": "Healthcare"},
    {"pattern": "dentist|odontoiatr", "category": "Healthcare"},
    {"pattern": "teach|docen|insegnan", "category": "Education"},
    {"pattern": "tutor(?!.*(?:clinical|medical))", "category": "Education"},
    {"pattern": "call\\s+center|customer\\s+(?:service|care)", "category": "Customer Service"},
]

updated = []
for pid, profile in data["profiles"].items():
    if not pid.startswith("msl-") and not pid.startswith("reg-"):
        continue
    bl = profile["filters"]["blacklist"]
    existing_pats = {p["pattern"] for p in bl}
    added = 0
    for item in extra_blacklist:
        if item["pattern"] not in existing_pats:
            bl.append(item)
            existing_pats.add(item["pattern"])
            added += 1
    if added:
        updated.append(f"{pid}: +{added} blacklist")

with open("profiles.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

for u in updated:
    print(f"  {u}")
print(f"\n✓ Aggiornati {len(updated)} profili")
