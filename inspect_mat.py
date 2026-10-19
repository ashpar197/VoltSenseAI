import numpy as np
from scipy.io import loadmat

d = loadmat("data/raw/B0005.mat", simplify_cells=True)
print("Top-level keys:", [k for k in d if not k.startswith("__")])

cycles = d["B0005"]["cycle"]
print("Total cycles:", len(cycles))

counts = {}
for c in cycles:
    counts[c["type"]] = counts.get(c["type"], 0) + 1
print("Cycle types:", counts)

first = next(c for c in cycles if c["type"] == "discharge")
print("Fields in the first discharge cycle:")
for name, value in first["data"].items():
    print("  ", name, "| length:", len(np.atleast_1d(value)))
