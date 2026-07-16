import csv

labels = {}
with open("data/anomaly_label.csv") as f:
    for row in csv.DictReader(f):
        labels[row["BlockId"]] = row["Label"]

train_normal = open("data/train_normal.txt", "w")
test_normal  = open("data/test_normal.txt",  "w")
test_anomaly = open("data/test_anomaly.txt", "w")
normal_count = 0

with open("data/sequences.tsv") as f:
    for line in f:
        parts = line.strip().split("\t", 1)
        if len(parts) < 2:
            continue
        blk, tokens = parts
        label = labels.get(blk, "Normal")
        if label == "Anomaly":
            test_anomaly.write(tokens + "\n")
        else:
            normal_count += 1
            if normal_count % 5 == 0:
                test_normal.write(tokens + "\n")
            else:
                train_normal.write(tokens + "\n")

train_normal.close(); test_normal.close(); test_anomaly.close()
print(f"Normal: {normal_count}  train ~{normal_count*4//5}  test ~{normal_count//5}")
print(f"Anomaly: {sum(1 for l in labels.values() if l == 'Anomaly')}")
