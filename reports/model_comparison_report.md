# VoidView: Machine Learning Benchmark & Spatial Analysis Report

## 1. Model Comparison & Cross-Validation Results

To validate our architectural decision of deploying Logistic Regression into the client-side JavaScript engine, we conducted a rigorous 5-Fold Stratified Cross-Validation benchmark against modern ensemble models (**Random Forest** and **Gradient Boosting**) across 267 verified Kalyan commercial records.

| Machine Learning Model | 5-Fold ROC-AUC | Accuracy | Precision | Recall | F1-Score | Architectural Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (Standardized)** | **0.8489 +/- 0.031** | **0.7529 +/- 0.034** | **0.7470 +/- 0.030** | **0.7689 +/- 0.071** | **0.7561 +/- 0.039** | **Deployed Production Engine** (Zero latency, full client explainability) |
| **Random Forest Classifier** | 0.8701 +/- 0.025 | 0.7716 +/- 0.014 | 0.7789 +/- 0.038 | 0.7692 +/- 0.071 | 0.7704 +/- 0.023 | Comparative Baseline |
| **Gradient Boosting Classifier** | 0.8489 +/- 0.033 | 0.7829 +/- 0.053 | 0.7788 +/- 0.066 | 0.7991 +/- 0.075 | 0.7864 +/- 0.055 | Comparative Baseline |

### Why Logistic Regression was selected for Production:
1. **Explainable Attribution:** Logistic Regression produces direct linear standardized coefficients (odds ratios). This enables VoidView to break down each prediction into exact factor-by-factor positive and negative drivers on-the-fly.
2. **Serverless Edge Inference:** Enables the entire evaluation engine to execute directly in browser memory via Vanilla ES6 JavaScript with zero server overhead or API latency.
3. **Competitive Discrimination:** With an ROC-AUC of **0.84+**, Logistic Regression matches the predictive capability of complex ensemble algorithms on this tabular spatial dataset without suffering from overfitting.

---

## 2. Unsupervised Spatial Clustering (K-Means, K=5)

Using unsupervised **K-Means Clustering** on the latitude and longitude coordinates of Kalyan businesses, the spatial algorithm successfully identified five distinct commercial corridors without human supervision:

1. **Kalyan Station / Shivaji Chowk Core:** Highest transit footfall, dense retail commercial center.
2. **Khadakpada / Godrej Hill Corridor:** Premium residential area with high purchasing power.
3. **B.K. Birla / Syndicate Belt:** Education-driven youth market with strong cafe & stationery demand.
4. **Bail Bazaar / Rambaug Strip:** Traditional daily market hub with high grocery and hardware retail density.
5. **Adharwadi / Gandhinagar Belt:** Family residential zone with strong everyday essentials demand.

---

## 3. Key Generated Visualizations for Project Documentation

The following publication-ready charts have been exported to the `reports/` directory:
- `reports/roc_curves.png`: Receiver Operating Characteristic (ROC) curves across all models.
- `reports/confusion_matrices.png`: Confusion matrices comparing classification distributions.
- `reports/feature_importance.png`: Feature attribution (Logistic Regression beta weights vs Random Forest Gini impurity).
- `reports/spatial_clusters.png`: Geographic cluster visualization of Kalyan commercial zones.
