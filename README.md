# VoltSense AI

Prototype that predicts EV battery State of Health (SOH) from NASA battery data.

## Data
NASA Battery Aging dataset: B0005, B0006, B0007, B0018 (636 discharge cycles).

## Method
- Target: SOH = capacity / first-cycle capacity x 100 (relative, not officially validated).
- Features: cycle, mean/min voltage, mean/max temperature per discharge.
- Model: Random Forest (100 trees), saved in models/voltsense.joblib.

## Results (hold-one-battery-out, SOH points of error)
- Cycle only, Linear: average 5.32
- Cycle + 4 features, Linear: average 3.75
- Cycle + 4 features, Forest: average 2.97 (2.7 to 4.6 per battery)
- Adding discharge duration made it worse (4.18) and was dropped.

## Limitations
- Only 4 lab batteries, so results may not carry over to real EVs.
- Final model is trained on all 4, so it has no unseen test battery.

## Run
.\.venv\Scripts\python.exe -m streamlit run .\app_gauge.py

## Rebuild from scratch
1. Put B0005.mat, B0006.mat, B0007.mat, B0018.mat in data\raw\
2. .\.venv\Scripts\python.exe .\prepare_data.py
3. .\.venv\Scripts\python.exe .\extract_features.py
4. .\.venv\Scripts\python.exe .\save_model.py
5. .\.venv\Scripts\python.exe -m streamlit run .\app_gauge.py
