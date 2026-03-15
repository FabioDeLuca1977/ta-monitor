import sqlite3, glob, pathlib

DBS = list(pathlib.Path('data').glob('*.db'))
for db_path in DBS:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
    if not cur.fetchone():
        conn.close()
        continue
    cur.execute("PRAGMA table_info(jobs)")
    cols = [c[1] for c in cur.fetchall()]
    if 'status' not in cols:
        cur.execute("ALTER TABLE jobs ADD COLUMN status TEXT DEFAULT NULL")
        cur.execute("ALTER TABLE jobs ADD COLUMN status_date TEXT DEFAULT NULL")
        conn.commit()
        print(f'OK: {db_path.name}')
    else:
        print(f'GIA OK: {db_path.name}')
    conn.close()
print('Migration completata.')
