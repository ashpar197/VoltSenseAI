# ⚡ VoltSense AI: EV Battery State of Health (SOH) & Life Prediction

VoltSense AI is a machine learning web application that predicts lithium-ion battery **State of Health (SOH)** and evaluates **Remaining Useful Life (RUL)** and second-life reusability using open laboratory cycling data from NASA.

---

## 📌 Project Overview

Electric Vehicle (EV) batteries degrade through repeated charge and discharge cycles, ambient temperature extremes, and usage patterns. VoltSense AI provides an interactive, data-driven dashboard allowing engineers, fleet managers, and researchers to:
1. **Predict Battery Health (SOH):** Estimate relative capacity degradation from key discharge metrics.
2. **Second-Life Suitability Assessment:** Classify pack condition (*Healthy*, *Aging*, or *Degraded*) and receive automated recommendations for stationary storage or recycling.
3. **Download PDF Diagnostic Reports:** Export a single-page assessment report with timestamps, inputs, predictions, and safety disclaimers.
4. **Inspect Fade Curves:** Visualize degradation curves across laboratory battery cells with an 80% SOH reference threshold.
5. **Estimate Remaining Useful Life (RUL):** Compute projected cycles to end-of-life (80% SOH) from historical degradation trends.
6. **Track & Compare Prediction History:** Store historical predictions permanently in a local SQLite database, compare multi-battery degradation curves, and export records to CSV.

---

## 🔬 Machine Learning Methodology

* **Target Variable:** Relative State of Health ($SOH_{rel}$):
  $$\text{SOH} = \frac{\text{Discharge Capacity (Ah)}}{\text{First Cycle Capacity (Ah)}} \times 100$$
* **Input Features (5 Features per Discharge Cycle):**
  * `cycle`: Discharge cycle number
  * `v_mean`: Mean voltage during discharge (V)
  * `v_min`: Minimum voltage reached at end-of-discharge (V)
  * `t_mean`: Mean cell temperature (°C)
  * `t_max`: Maximum cell temperature reached (°C)
* **Model Architecture:** **Random Forest Regressor** (100 estimators, fixed random state for reproducibility).
* **Validation & Performance:**
  * Evaluated via Leave-One-Battery-Out (hold-out) cross-validation on NASA cells.
  * Achieved an average error of approximately ~3 SOH points.
  * Model serialized and loaded from `models/voltsense.joblib`.

---

## 📊 Dataset Information

VoltSense AI utilizes the **NASA Prognostics Center of Excellence (PCoE) Battery Aging Dataset**:
* **Selected Cells:** `B0005`, `B0006`, `B0007`, and `B0018`.
* **Test Conditions:** Cycled under continuous 2A discharge profiles at room temperature (~24°C) down to fixed cut-off voltages (2.7V, 2.5V, 2.2V).
* **Processed Dataset:** `data/processed/battery_features.csv` (636 processed discharge cycles).

---

## 🛠️ Technologies Used

* **Language:** Python 3.11+
* **Frontend / Framework:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** [scikit-learn](https://scikit-learn.org/), [joblib](https://joblib.readthedocs.io/)
* **Data Processing:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Visualization:** [Altair](https://altair-viz.github.io/), [Matplotlib](https://matplotlib.org/)
* **Database:** SQLite3 (built-in standard library)
* **Deployment Platform:** Streamlit Community Cloud

---

## 📁 Project Structure

```text
VoltSenseAI/
├── .gitignore                      # Git ignore configuration
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Production Python dependencies
├── app_all.py                      # Main Streamlit application entry point (4 tabs)
├── database.py                     # SQLite database storage & query module
├── pdf_report.py                   # High-resolution PDF assessment report generator
├── recommend.py                    # Second-life rule-based classification engine
│
├── data/
│   ├── processed/
│   │   ├── battery_capacity.csv    # Extracted cycle-by-cycle capacity
│   │   └── battery_features.csv    # Extracted voltage, temperature & cycle features
│   └── raw/                        # Original NASA .mat files (gitignored)
│
├── models/
│   └── voltsense.joblib            # Trained Random Forest model bundle
│
├── extract_features.py             # Feature engineering pipeline
├── prepare_data.py                 # Raw MATLAB .mat parser
└── save_model.py                   # Model training and export script
```

---

## 💻 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/VoltSenseAI.git
cd VoltSenseAI
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app_all.py
```
Open your browser at **http://localhost:8501**.

---

## ☁️ Streamlit Community Cloud Deployment Guide

Follow these steps to deploy VoltSense AI to the cloud for free:

### Step 1: Push Code to GitHub
1. Stage and commit your files:
   ```bash
   git add .
   git commit -m "Prepare VoltSense AI for Streamlit Cloud deployment"
   ```
2. Push to your GitHub repository:
   ```bash
   git branch -M main
   git push -u origin main
   ```
   *(Ensure `models/voltsense.joblib` and `data/processed/battery_features.csv` are included in the repository!)*

### Step 2: Deploy on Streamlit Cloud
1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **"New app"**.
3. Select your GitHub repository (`<username>/VoltSenseAI`).
4. Set **Branch** to `main`.
5. Set **Main file path** to:
   ```text
   app_all.py
   ```
6. Click **"Deploy!"**.

Streamlit Cloud will automatically detect `requirements.txt`, install dependencies, and launch the web app.

---

## ⚠️ Scientific Limitations & Safety Disclaimer

1. **Laboratory Prototype:** VoltSense AI was developed using 4 laboratory-tested cells under fixed current discharges. Results may vary on real-world EV battery packs subject to regenerative braking, variable current draws, and cell-balancing dynamics.
2. **Decision Support Only:** Predicted SOH and remaining life metrics are mathematical approximations and do not constitute a formal guarantee of electrical safety or pack integrity. Professional physical inspection and BMS diagnostics are mandatory before repurposing any lithium-ion battery.
