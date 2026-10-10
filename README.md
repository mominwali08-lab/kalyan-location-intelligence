# 🌌 VoidView: AI-Powered Retail Location Intelligence System
> *"From void to viability."*

An AI-driven geospatial decision-support system designed to evaluate and predict retail site viability across Kalyan City (Maharashtra, India).

🌐 **Live Web Application:** [https://mominwali08-lab.github.io/VoidView/](https://mominwali08-lab.github.io/VoidView/)

---

## 📌 Project Overview
When opening a new brick-and-mortar retail business in Kalyan, choosing the right location is the single most critical factor for survival. **VoidView** eliminates the guesswork by combining machine learning with geospatial analytics:

- **Instant Viability Score (0–100%):** Evaluates any point clicked on the interactive map or searched by landmark.
- **Factor-by-Factor Explainability:** Transparent breakdown of positive drivers and risk factors (e.g. competitor counts, distance to closest rival, commercial density, walking footfall).
- **Spatial Intelligence Layer:**
  - **Pedestrian Footfall Index (0–100):** Weighted proximity to Kalyan's 6 primary transit and commercial nodes.
  - **Anchor Tenant Synergy Engine:** Proximity boosts (+8% to +30%) from 13 critical anchors (hospitals, colleges, transit hubs).
  - **Demographic Zone Multipliers:** Purchasing power and audience targeting across 6 distinct Kalyan neighborhoods.
- **Comparative Category Analysis:** Ranks all 10 retail categories for any chosen spot.
- **Smart Location Advice:** Strategic guidance for risky locations (suggesting nearby directions or optimal business types).

---

## 🛠️ Project Repository Structure

```text
VoidView/
├── index.html                           # Live Web Application (Leaflet.js + Vanilla JS)
├── VoidView_Data_Science_Benchmark.ipynb # Complete Data Science Notebook (Models, 5-Fold CV, Plots)
├── requirements.txt                     # Python dependencies
├── data/
│   ├── kalyan_businesses.csv            # Dataset of 267 verified businesses from Google Places API
│   └── new_model_weights.json           # Exported Logistic Regression weights & feature scalers
├── reports/
│   ├── roc_curves.png                   # ROC curve comparison across 3 ML models
│   ├── confusion_matrices.png           # Confusion matrices for all models
│   ├── feature_importance.png           # Feature attribution (Beta weights vs Gini importance)
│   ├── spatial_clusters.png             # K-Means spatial cluster map (K=5)
│   └── model_comparison_report.md       # Full benchmark performance table & analysis
├── REPORT_NOTES.md                      # Comprehensive methodology notes & viva preparation guide
└── README.md                            # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Web Application (No Setup Needed)
- Open the live deployment: **[https://mominwali08-lab.github.io/VoidView/](https://mominwali08-lab.github.io/VoidView/)**
- Or locally: double-click `index.html` to run in any browser.

### 2. Data Science Notebook & Benchmarks
To view or re-run the machine learning models and visualizations:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open the Jupyter Notebook
jupyter notebook VoidView_Data_Science_Benchmark.ipynb
```

---

## 📊 Model Performance Summary (5-Fold Stratified CV)

| Machine Learning Model | 5-Fold ROC-AUC | Accuracy | F1-Score | Role in Architecture |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression (Standardized)** | **0.8489 ± 0.031** | **75.29%** | **0.7561** | **Deployed Production Engine** (Zero latency, full factor-level explainability) |
| **Random Forest Classifier** | 0.8701 ± 0.025 | 77.16% | 0.7704 | Comparative Baseline |
| **Gradient Boosting Classifier** | 0.8489 ± 0.033 | 78.29% | 0.7864 | Comparative Baseline |
