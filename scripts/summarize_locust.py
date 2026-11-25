"""Turn Locust CSV results (<N>_containers_stats.csv) into a Markdown table for the README.

Usage: python scripts/summarize_locust.py results/locust
"""
import csv
import re
import sys
from pathlib import Path

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results/locust")
rows = []
for f in sorted(folder.glob("*_containers_stats.csv"), key=lambda p: int(re.match(r"\d+", p.name).group())):
    n = int(re.match(r"\d+", f.name).group())
    stats = {r["Name"]: r for r in csv.DictReader(f.open())}
    for name in ("/predict", "Aggregated"):
        r = stats.get(name)
        if r:
            reqs, fails = int(r["Request Count"]), int(r["Failure Count"])
            rows.append([n, name, reqs, f"{100 * fails / reqs:.1f}%" if reqs else "-",
                         f"{float(r['Requests/s']):.1f}", r["50%"], f"{float(r['Average Response Time']):.0f}",
                         r["95%"], r["99%"], r["Max Response Time"].split(".")[0]])

header = ["Containers", "Endpoint", "Requests", "Failures", "Req/s", "Median (ms)", "Mean (ms)",
          "95th pct (ms)", "99th pct (ms)", "Max (ms)"]
lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
table = "\n".join(lines)
(folder / "summary.md").write_text(table + "\n")
print(table)
