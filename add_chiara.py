#!/usr/bin/env python3
import json, hashlib

with open("users.json", "r", encoding="utf-8") as f:
    data = json.load(f)

pw = "ChiaraSisti"
pw_hash = hashlib.sha256(pw.encode()).hexdigest()

if any(u["username"] == "chiarasisti" for u in data["users"]):
    print("SKIP: utente chiarasisti esiste già")
else:
    data["users"].append({
        "username": "chiarasisti",
        "password_hash": pw_hash,
        "email": "chiarasisti@libero.it",
        "role": "user",
        "created": "2026-03-08"
    })
    print("✓ Aggiunto: chiarasisti (chiarasisti@libero.it)")

with open("users.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Utenti totali: {[u['username'] for u in data['users']]}")
