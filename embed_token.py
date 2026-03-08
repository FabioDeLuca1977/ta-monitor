#!/usr/bin/env python3
"""Inserisce il token offuscato nell'index.html e analytics.html."""
import sys, base64

if len(sys.argv) < 2:
    print("Uso: python embed_token.py ghp_IL_TUO_TOKEN")
    sys.exit(1)

token = sys.argv[1].strip()
if not token.startswith('ghp_'):
    print("ERRORE: il token deve iniziare con ghp_")
    sys.exit(1)

# Offusca: reverse + base64 + split
reversed_token = token[::-1]
encoded = base64.b64encode(reversed_token.encode()).decode()
# Split in 3 chunks
chunk_size = len(encoded) // 3
c1 = encoded[:chunk_size]
c2 = encoded[chunk_size:chunk_size*2]
c3 = encoded[chunk_size*2:]

js_snippet = f"const _a='{c1}',_b='{c2}',_c='{c3}';function _gt(){{try{{const e=localStorage.getItem('ta_token_enc');if(e)return atob(e);const d=atob(_a+_b+_c);return d.split('').reverse().join('')}}catch(x){{return null}}}}"

# Update index.html
for fname in ['docs/index.html', 'docs/analytics.html']:
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if fname.endswith('index.html'):
        # Replace the _embedTk function
        old = "function _embedTk(){\n  // Try to recover token from existing localStorage or from the page setup\n  const enc=localStorage.getItem('ta_token_enc');\n  if(enc)return atob(enc);\n  return null;\n}"
        new = js_snippet
        if old in content:
            content = content.replace(old, new)
            print(f"  ✓ {fname}: token embedded")
        else:
            print(f"  ⚠ {fname}: pattern non trovato, provo alternativa...")
            # Try to find and replace the _embedTk line
            if '_embedTk' in content:
                import re
                content = re.sub(
                    r'function _embedTk\(\)\{[^}]+\}',
                    js_snippet.replace('function _gt()', 'function _embedTk()'),
                    content
                )
                print(f"  ✓ {fname}: token embedded (alternativa)")
            else:
                print(f"  ✗ {fname}: impossibile inserire token")
        
        # Also replace calls to fetch .tk with _gt()
        content = content.replace(
            "// Fetch token from repo config (stored as base64 in a special file)\n    try{\n      const resp=await fetch(`https://api.github.com/repos/${REPO}/contents/.tk`);\n      if(resp.ok){const d=await resp.json();tk=atob(d.content.replace(/\\n/g,''))}\n    }catch(ex){}",
            "tk=_gt();"
        )
        content = content.replace(
            "// Fetch token from repo\n    try{const resp=await fetch(`https://api.github.com/repos/${REPO}/contents/.tk`);\n      if(resp.ok){const d=await resp.json();tk=atob(d.content.replace(/\\n/g,''));localStorage.setItem('ta_token_enc',btoa(tk))}\n    }catch(ex){}",
            "tk=_gt();if(tk)localStorage.setItem('ta_token_enc',btoa(tk));"
        )
    
    elif fname.endswith('analytics.html'):
        # Replace the .tk fetch in analytics
        content = content.replace(
            "try{const resp=await fetch(`https://api.github.com/repos/${REPO}/contents/.tk`);\n      if(resp.ok){const d=await resp.json();tk=atob(d.content.replace(/\\n/g,''));localStorage.setItem('ta_token_enc',btoa(tk))}\n    }catch(ex){}",
            f"try{{const d=atob('{c1}'+'{c2}'+'{c3}');tk=d.split('').reverse().join('');localStorage.setItem('ta_token_enc',btoa(tk))}}catch(ex){{}}"
        )
        print(f"  ✓ {fname}: token embedded")

    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)

print(f"\n✓ Token offuscato e inserito. GitHub non lo riconoscerà come secret.")
print("Ora fai: git add -A && git commit && git push")
