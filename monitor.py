# monitor.py

from collections import Counter

log_file = "monitor.log"

if not os.path.exists(log_file):
    print("No monitoring data yet.")
    exit()

# Read log file
with open(log_file, "r") as f:
    lines = f.readlines()

predictions = [int(line.split("Predicted: ")[1].split(",")[0]) for line in lines]

# Count predictions
counter = Counter(predictions)

print("Monitoring Results:")
for cls, count in counter.items():
    print(f"Class {cls}: {count} predictions")
