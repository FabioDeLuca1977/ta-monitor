#!/usr/bin/env python3
"""Fix blacklist per falsi positivi residui."""
import json

with open("profiles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

extra = [
    {"pattern": "speech\\s+therapist|logopedist", "category": "Healthcare"},
    {"pattern": "battery|vanadium|redox|flow\\s+energy", "category": "Engineering"},
    {"pattern": "BMS\\s+engineer", "category": "Engineering"},
    {"pattern": "power\\s+electronics", "category": "Engineering"},
    {"pattern": "stack\\s+scaling", "category": "Engineering"},
    {"pattern": "\\bengineering\\b(?!.*(?:medical|pharma|clinical|regulatory))", "category": "Engineering"},
    {"pattern": "recreation\\s+aid", "category": "Non pertinente"},
    {"pattern": "CYS\\s+assistant", "category": "Non pertinente"},
    {"pattern": "NF-\\d+", "category": "Non pertinente"},
]

for pid, profile in data["profiles"].items():
    if not pid.startswith("msl-") and not pid.startswith("reg-"):
        continue
    bl = profile["filters"]["blacklist"]
    existing = {p["pattern"] for p in bl}
    added = 0
    for item in extra:
        if item["pattern"] not in existing:
            bl.append(item)
            existing.add(item["pattern"])
            added += 1
    if added:
        print(f"  {pid}: +{added} blacklist (totale: {len(bl)})")

with open("profiles.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✓ Done")
