# Location Intelligence System for Kalyan (Maharashtra, India)

A machine learning decision-support system for commercial site selection in Kalyan. Designed as an honest, defendable final-year capstone project built with free, open tools and real Google Places data.

---

## 📌 Project Overview
When opening a new business in Kalyan, picking the right spot is critical. A user selects a geographic spot (via address search or clicking on the map) and chooses a business type. The application returns:
- A location viability percentage score.
- Plain-English breakdown of the top factors influencing that score (based on linear model coefficients).
- Nearby direct competitors overlaid on a dark Leaflet map.
- If the spot is **Risky (<50%)**: suggests nearby compass directions (at 200m and 400m) with higher viability, and identifies business categories that would thrive at the chosen spot.
- Complete ranked viability comparison across all 10 business types.

---

## 🛠️ Project Structure
```text
kalyan-location-intelligence/
├── data/
│   └── kalyan_businesses.csv       # Original dataset (267 real Google Places businesses)
├── index.html                      # Offline Dark-Theme Web Application (Runs in any browser)
├── app.py                          # All-in-one Streamlit Python Web Application
├── requirements.txt                # Python package dependencies
├── REPORT_NOTES.md                 # Full project report, methodology, and 10 viva answers
└── README.md                       # Project documentation & run guide
```

---

## 💻 How to Run the Project

### Method 1: The Offline Web Application (Recommended)
No terminal or Python installation is required.

1. Open your File Explorer to this folder.
2. **Double-click `index.html`**.
3. It opens directly in **Google Chrome**, **Microsoft Edge**, or your default web browser.

---

### Method 2: The Python / Streamlit Version
If your professor or examiner asks to see the Python code running in a terminal:

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Run the application:
   ```powershell
   python -m streamlit run app.py
   ```
3. Open your browser at `http://localhost:8501`.

---

## 📚 Viva Defense & Documentation
- For detailed methodology, cross-validation metrics, feature ablation tables, and **10 prepared viva questions and answers**, see **[`REPORT_NOTES.md`](REPORT_NOTES.md)**.
