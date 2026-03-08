#!/usr/bin/env python3
"""Inserisce la Gemini API key nell'index.html."""
import sys

if len(sys.argv) < 2:
    print("Uso: python embed_gemini.py LA_TUA_GEMINI_API_KEY")
    print("\nPer ottenere la key gratuita:")
    print("1. Vai su https://aistudio.google.com/apikey")
    print("2. Clicca 'Create API key'")
    print("3. Copia la key (inizia con 'AIza...')")
    sys.exit(1)

key = sys.argv[1].strip()

with open("docs/index.html", "r", encoding="utf-8") as f:
    content = f.read()

if '%%GEMINI_KEY%%' in content:
    content = content.replace('%%GEMINI_KEY%%', key)
    print(f"✓ Gemini API key inserita in docs/index.html")
elif key in content:
    print("Key già presente in docs/index.html")
else:
    print("⚠ Placeholder %%GEMINI_KEY%% non trovato — key potrebbe essere già stata inserita")

with open("docs/index.html", "w", encoding="utf-8") as f:
    f.write(content)
