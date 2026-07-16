import csv
import re
import sys
from collections import defaultdict

BLK_RE = re.compile(r"(blk_-?\d+)")
sequences = defaultdict(list)

with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        m = BLK_RE.search(row["Content"])
        if not m:
            continue
        sequences[m.group(1)].append(row["EventId"])

for blk, events in sequences.items():
    print(f"{blk}\t{' '.join(events)}")
