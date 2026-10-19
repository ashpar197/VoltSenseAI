"""
prepare_data.py - VoltSense AI data preparation.

Reads the NASA Battery Aging .mat files in data/raw/, extracts the discharge
cycles, and writes data/processed/battery_capacity.csv.

Columns:
    battery_id                 e.g. "B0005"
    cycle                      sequential discharge-cycle number (1, 2, 3, ...)
    capacity_ah                measured discharge capacity in Ah (from the file)
    soh_percent_provisional    capacity_ah / NOMINAL_CAPACITY_AH * 100

ASSUMPTION TO VALIDATE: the dataset documentation states a nominal capacity
of 2.0 Ah, so SOH is provisionally computed against that value. This is NOT
yet validated. Alternatives (e.g. using each battery's own first-cycle
capacity as 100%) should be compared before this is used as a final target.

Nothing is invented: every capacity comes straight from the .mat files.
Records with missing or invalid capacity are skipped and reported, never
filled in.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "battery_capacity.csv"

BATTERY_IDS = ["B0005", "B0006", "B0007", "B0018"]
NOMINAL_CAPACITY_AH = 2.0  # provisional assumption, see docstring


def extract_discharge_rows(battery_id):
    """Return (rows, skipped_messages) for one battery."""
    path = RAW_DIR / f"{battery_id}.mat"
    mat = loadmat(path, simplify_cells=True)

    if battery_id not in mat:
        raise KeyError(
            f"{path.name}: expected top-level key '{battery_id}', "
            f"found {[k for k in mat if not k.startswith('__')]}"
        )

    cycles = mat[battery_id]["cycle"]
    rows, skipped = [], []
    discharge_number = 0

    for record in cycles:
        if record["type"] != "discharge":
            continue
        # Count every discharge record so numbering reflects the true order,
        # even if an invalid one is skipped (that leaves a visible gap).
        discharge_number += 1

        try:
            capacity = np.asarray(record["data"]["Capacity"], dtype=float)
        except (KeyError, TypeError, ValueError) as err:
            skipped.append(f"{battery_id} cycle {discharge_number}: "
                           f"no usable Capacity ({err})")
            continue

        if capacity.size != 1:
            skipped.append(f"{battery_id} cycle {discharge_number}: "
                           f"Capacity has {capacity.size} values, expected 1")
            continue

        capacity = float(capacity.reshape(-1)[0])
        if not np.isfinite(capacity) or capacity <= 0:
            skipped.append(f"{battery_id} cycle {discharge_number}: "
                           f"invalid capacity {capacity}")
            continue

        rows.append({
            "battery_id": battery_id,
            "cycle": discharge_number,
            "capacity_ah": capacity,
        })

    return rows, skipped


def main():
    # 1. Check that every expected file exists before doing anything.
    missing = [b for b in BATTERY_IDS if not (RAW_DIR / f"{b}.mat").is_file()]
    if missing:
        print("ERROR: missing file(s) in", RAW_DIR)
        for b in missing:
            print(f"  - {b}.mat")
        print("Nothing was written. Copy the files into data/raw/ and re-run.")
        sys.exit(1)

    # 2. Extract discharge rows from each battery.
    all_rows, all_skipped = [], []
    for battery_id in BATTERY_IDS:
        try:
            rows, skipped = extract_discharge_rows(battery_id)
        except Exception as err:
            print(f"ERROR while reading {battery_id}.mat: {err}")
            print("Nothing was written.")
            sys.exit(1)
        all_rows.extend(rows)
        all_skipped.extend(skipped)

    if not all_rows:
        print("ERROR: no valid discharge records found. Nothing was written.")
        sys.exit(1)

    # 3. Build the table and the provisional SOH column.
    df = pd.DataFrame(all_rows)
    df["soh_percent_provisional"] = df["capacity_ah"] / NOMINAL_CAPACITY_AH * 100

    # 4. Save (this CSV is a generated file; re-running regenerates it).
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # 5. Report.
    print(f"Saved {len(df)} records to {OUT_PATH}\n")
    print("Records per battery:")
    print(df.groupby("battery_id").size().to_string(), "\n")
    if all_skipped:
        print(f"WARNING: {len(all_skipped)} record(s) skipped:")
        for msg in all_skipped:
            print("  -", msg)
        print()
    print("Preview:")
    print(df.head().to_string(index=False))
    print("\nNOTE: soh_percent_provisional assumes a 2.0 Ah nominal capacity "
          "(initial assumption, not yet validated).")


if __name__ == "__main__":
    main()
