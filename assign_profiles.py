#!/usr/bin/env python3
"""Assegna i profili MSL e Regulatory a gabriellabellavia."""
import json

with open("profiles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

changed = []
for pid, profile in data["profiles"].items():
    if pid.startswith("msl-") or pid.startswith("reg-"):
        profile["created_by"] = "gabriellabellavia"
        changed.append(pid)
        print(f"  {pid} → created_by: gabriellabellavia")

with open("profiles.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✓ {len(changed)} profili assegnati a gabriellabellavia")
