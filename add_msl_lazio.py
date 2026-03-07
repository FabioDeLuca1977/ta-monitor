#!/usr/bin/env python3
"""Aggiunge il profilo Medical Liaison Lazio al profiles.json."""
import json

with open("profiles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

if "msl-lazio" in data["profiles"]:
    print("SKIP: msl-lazio esiste già")
else:
    # Copy filters from msl-veneto (same structure)
    msl = json.loads(json.dumps(data["profiles"]["msl-veneto"]))
    msl["name"] = "Medical Liaison — Lazio"
    msl["description"] = "Monitoraggio posizioni MSL, Medical Affairs e ruoli scientifici pharma nel Lazio"
    msl["location"] = "Roma, Lazio, Italia"
    msl["created"] = "2026-03-07T00:00:00"
    data["profiles"]["msl-lazio"] = msl
    print("Aggiunto: msl-lazio")

with open("profiles.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Profili totali: {list(data['profiles'].keys())}")
