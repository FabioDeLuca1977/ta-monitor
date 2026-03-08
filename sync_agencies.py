#!/usr/bin/env python3
"""
Sincronizza le agenzie su tutti i profili.

Uso:
  python sync_agencies.py                          → mostra stato attuale
  python sync_agencies.py --add "NuovaAgenzia"     → aggiunge a tutti i profili
  python sync_agencies.py --add "Agenzia1,Agenzia2" → aggiunge multiple
  python sync_agencies.py --remove "VecchiaAgenzia" → rimuove da tutti
  python sync_agencies.py --sync                    → allinea tutti i profili al set completo
"""
import json, argparse, sys

PROFILES_FILE = "profiles.json"

def load():
    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save(data):
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_all_agencies(data):
    """Raccoglie tutte le agenzie uniche da tutti i profili."""
    all_ag = set()
    for pid, p in data.get("profiles", {}).items():
        for a in p.get("filters", {}).get("agencies", []):
            all_ag.add(a)
    return sorted(all_ag)

def show_status(data):
    all_ag = get_all_agencies(data)
    print(f"\n📊 Stato agenzie — {len(all_ag)} agenzie totali\n")
    for pid, p in data.get("profiles", {}).items():
        ag = p.get("filters", {}).get("agencies", [])
        missing = set(all_ag) - set(ag)
        status = "✓" if not missing else f"⚠ mancano {len(missing)}"
        print(f"  {pid}: {len(ag)} agenzie {status}")
        if missing:
            for m in sorted(missing):
                print(f"    - {m}")
    print(f"\n  Elenco completo ({len(all_ag)}):")
    for i, a in enumerate(all_ag, 1):
        print(f"    {i:2}. {a}")

def add_agencies(data, new_list):
    """Aggiunge agenzie a TUTTI i profili."""
    added_total = 0
    for pid, p in data.get("profiles", {}).items():
        filters = p.get("filters", {})
        existing = set(filters.get("agencies", []))
        to_add = [a for a in new_list if a not in existing]
        if to_add:
            filters["agencies"] = sorted(existing | set(new_list))
            added_total += len(to_add)
            print(f"  {pid}: +{len(to_add)} → {', '.join(to_add)}")
        else:
            print(f"  {pid}: già presenti")
    save(data)
    print(f"\n✓ Aggiunte {len(new_list)} agenzie a tutti i profili")

def remove_agencies(data, rem_list):
    """Rimuove agenzie da TUTTI i profili."""
    for pid, p in data.get("profiles", {}).items():
        filters = p.get("filters", {})
        before = len(filters.get("agencies", []))
        filters["agencies"] = sorted([a for a in filters.get("agencies", []) if a not in rem_list])
        after = len(filters["agencies"])
        removed = before - after
        if removed:
            print(f"  {pid}: -{removed}")
        else:
            print(f"  {pid}: nessuna da rimuovere")
    save(data)
    print(f"\n✓ Rimosse {len(rem_list)} agenzie da tutti i profili")

def sync_all(data):
    """Allinea tutti i profili al set completo di agenzie."""
    all_ag = get_all_agencies(data)
    synced = 0
    for pid, p in data.get("profiles", {}).items():
        filters = p.get("filters", {})
        existing = set(filters.get("agencies", []))
        missing = set(all_ag) - existing
        if missing:
            filters["agencies"] = sorted(existing | set(all_ag))
            synced += len(missing)
            print(f"  {pid}: +{len(missing)} agenzie sincronizzate")
        else:
            print(f"  {pid}: ✓ già allineato")
    save(data)
    print(f"\n✓ Sincronizzati tutti i profili ({len(all_ag)} agenzie)")

def main():
    parser = argparse.ArgumentParser(description="Gestione agenzie su tutti i profili")
    parser.add_argument("--add", help="Agenzie da aggiungere (separate da virgola)")
    parser.add_argument("--remove", help="Agenzie da rimuovere (separate da virgola)")
    parser.add_argument("--sync", action="store_true", help="Sincronizza tutti i profili al set completo")
    args = parser.parse_args()

    data = load()

    if args.add:
        new_list = [a.strip() for a in args.add.split(",") if a.strip()]
        print(f"\n➕ Aggiunta: {', '.join(new_list)}")
        add_agencies(data, new_list)
    elif args.remove:
        rem_list = [a.strip() for a in args.remove.split(",") if a.strip()]
        print(f"\n➖ Rimozione: {', '.join(rem_list)}")
        remove_agencies(data, rem_list)
    elif args.sync:
        print("\n🔄 Sincronizzazione agenzie...")
        sync_all(data)
    else:
        show_status(data)

if __name__ == "__main__":
    main()
