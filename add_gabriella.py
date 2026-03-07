#!/usr/bin/env python3
"""Aggiunge l'account di Gabriella a users.json."""
import json, hashlib

with open("users.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Hash password
pw = "Jobspy2026!"
pw_hash = hashlib.sha256(pw.encode()).hexdigest()

# Check if already exists
if any(u["username"] == "gabriellabellavia" for u in data["users"]):
    print("SKIP: utente gabriellabellavia esiste già")
else:
    data["users"].append({
        "username": "gabriellabellavia",
        "password_hash": pw_hash,
        "email": "gabriellabellavia@hotmail.it",
        "role": "user",
        "created": "2026-03-07"
    })
    print("✓ Aggiunto: gabriellabellavia (gabriellabellavia@hotmail.it)")

with open("users.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Utenti totali: {[u['username'] for u in data['users']]}")
