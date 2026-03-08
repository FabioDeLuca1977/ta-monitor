#!/usr/bin/env python3
"""Aggiunge le agenzie di default a tutti i profili che non le hanno."""
import json

DEFAULT_AGENCIES = [
    'Adami & Associati', 'Adecco', 'Ali Lavoro', 'Altaïde', 'Alti Profili',
    'Amrop', 'Antal', 'Articolo1', 'Badenoch + Clark', 'Boyden',
    'Carter & Benson', 'Catenon', 'e-work', 'Egon Zehnder', 'Etjca',
    'Experis', 'Gi Group', 'Grafton', 'Hays', 'Heidrick & Struggles',
    'Humangest', 'Hunters Group', 'In Job', 'IQM Selezione', 'Kelly Services',
    'Key2People', 'Korn Ferry', 'Lavoropiù', 'MAW', 'Manpower',
    'MF Consultant', 'Michael Page', 'Odgers Berndtson', 'Openjobmetis',
    'Orienta', 'Page Personnel', 'PharmaPoint', 'Quanta', 'Randstad',
    'Robert Half', 'Robert Walters', 'Russell Reynolds', 'S&you',
    'Spencer Stuart', 'Spring Professional', 'Stryker', 'Synergie',
    'Temporary', 'Umana', 'W Executive'
]

with open("profiles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for pid, profile in data.get("profiles", {}).items():
    filters = profile.get("filters", {})
    existing = set(filters.get("agencies", []))
    new_agencies = sorted(set(DEFAULT_AGENCIES) | existing)
    filters["agencies"] = new_agencies
    added = len(new_agencies) - len(existing)
    print(f"  {pid}: {len(existing)} → {len(new_agencies)} agenzie (+{added})")

with open("profiles.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✓ Aggiornati {len(data['profiles'])} profili con {len(DEFAULT_AGENCIES)} agenzie di default")
