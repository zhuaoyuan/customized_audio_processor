import sqlite3, json, sys
c = sqlite3.connect(r"out/jobs.db")
c.row_factory = sqlite3.Row
print("TABLES:", [r[0] for r in c.execute("select name from sqlite_master where type='table'")])
for r in c.execute("select * from jobs order by rowid desc limit 10"):
    d = dict(r)
    print(json.dumps(d, ensure_ascii=False, default=str))
print("--- stages ---")
try:
    for r in c.execute("select * from stages order by rowid desc limit 30"):
        print(json.dumps(dict(r), ensure_ascii=False, default=str))
except Exception as e:
    print("stages err", e)
