"""
Read preprocessed/Event_traces.csv (already grouped by block, labels embedded).
No parse step or anomaly_label.csv join needed.

Output:
  data/train_normal.txt  — 80% of Success sequences, one per line
  data/test_normal.txt   — 20% of Success sequences
  data/test_anomaly.txt  — all Fail sequences
"""
import csv
import re

BLK_RE = re.compile(r"E\d+")

train_normal, test_normal, test_anomaly = [], [], []
normal_count = 0

with open("data/preprocessed/Event_traces.csv", newline="") as f:
    for row in csv.DictReader(f):
        events = BLK_RE.findall(row["Features"])
        if not events:
            continue
        if row["Label"] == "Fail":
            test_anomaly.append(" ".join(events))
        else:
            normal_count += 1
            if normal_count % 5 == 0:
                test_normal.append(" ".join(events))
            else:
                train_normal.append(" ".join(events))

open("data/train_normal.txt", "w").write("\n".join(train_normal) + "\n")
open("data/test_normal.txt",  "w").write("\n".join(test_normal)  + "\n")
open("data/test_anomaly.txt", "w").write("\n".join(test_anomaly) + "\n")

print(f"train_normal: {len(train_normal)}")
print(f"test_normal:  {len(test_normal)}")
print(f"test_anomaly: {len(test_anomaly)}")
