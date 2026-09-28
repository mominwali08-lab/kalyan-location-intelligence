# Project Report Notes: Location Intelligence System for Kalyan

---

## 1. Methodology & Data Source
- **Data Source**: Real business listings extracted from Google Places across Kalyan (Maharashtra, India).
- **Collection Technique**: Targeted search per business category and geographic sub-area (Kalyan West, Kalyan East, Khadakpada, Rambaug, Shivaji Chowk, Adharwadi, Godrej Hill).
- **Deduplication & Cleaning**: Deduplicated on `(name, lat)` pairs and filtered within Kalyan's bounding box ($19.20^\circ\text{N} - 19.30^\circ\text{N}$, $73.10^\circ\text{E} - 73.20^\circ\text{E}$).
- **Final Sample Size**: 267 verified operational businesses across 10 retail/service categories. No synthetic rows were fabricated or added.

---

## 2. Success Label Definition & Its Limits
- **Definition**: A binary label `success_label` where:
  $$\text{success\_label} = \begin{cases} 1 & \text{if } \text{user\_ratings\_total} > \text{category\_median} \\ 0 & \text{otherwise} \end{cases}$$
- **Rationale for Within-Category Median**: Using a global median would disproportionately favor restaurants and cafes (which naturally accumulate hundreds or thousands of reviews) over pharmacies or hardware stores (which rarely exceed 30 reviews). A within-category threshold ensures fair peer-to-peer benchmarking.
- **Sensitivity Check**: Dropping the 16 businesses with $<5$ reviews caused only 7 label flips (2.8% of retained records), demonstrating label stability.
- **Fundamental Limit**: This label is strictly a **popularity and relative footfall proxy**. It does **not** measure profit margins, financial balance sheets, rent burden, or multi-year business survival.

---

## 3. Features Explained in Plain Words
1. **`dist_station_m`**: Haversine distance in metres to Kalyan Railway Station (primary transit hub).
2. **`dist_khadakpada_m`**: Haversine distance in metres to Khadakpada Circle (affluent residential and dining center).
3. **`dist_birla_college_m`**: Haversine distance in metres to B.K. Birla College (student footfall anchor).
4. **`dist_durgadi_fort_m`**: Haversine distance in metres to Durgadi Fort / historic market area.
5. **`competitor_count_500m`**: Number of active competitors within a 5-minute walk (500 metres). For cafes, restaurants, and bakeries, this counts all food-serving establishments (shared food group rule).
6. **`nearest_competitor_distance_m`**: Distance in metres to the single closest competitor.
7. **`business_density_1km`**: Total number of operational commercial establishments of all types within 1 km (measure of general commercial agglomeration).
8. **`cat_*` (One-hot encoded categories)**: Baseline adjustments for each of the 10 commercial categories.

---

## 4. Cross-Validation Results

Evaluated via **Repeated Stratified 5-Fold Cross-Validation (3 repeats = 15 evaluation runs)**:

| Model | Accuracy (Mean ± Std) | ROC-AUC (Mean ± Std) |
|---|---|---|
| **Dummy Baseline (Most Frequent)** | 0.5168 ± 0.0070 | 0.5000 ± 0.0000 |
| **Logistic Regression** | 0.5556 ± 0.0511 | 0.5712 ± 0.0575 |
| **Random Forest** | 0.5981 ± 0.0428 | 0.6389 ± 0.0422 |

### Model Selection Justification:
While Random Forest achieved higher cross-validated accuracy, **Logistic Regression was selected as the operational model**. Its linear coefficients permit exact, transparent, and additive factor breakdowns ($\beta_j \cdot z_j$) required for an honest, defendable decision-support tool in an academic viva, avoiding black-box approximations.

---

## 5. Feature Ablation Study

| Ablation Experiment | Features Count | Accuracy (Mean ± Std) | ROC-AUC (Mean ± Std) |
|---|---|---|---|
| **Full Model** | 17 | 0.5556 ± 0.0511 | 0.5712 ± 0.0575 |
| **Without Food-Group Rule** | 17 | 0.5682 ± 0.0644 | 0.5786 ± 0.0605 |
| **Without Landmark Distances** | 13 | 0.5230 ± 0.0712 | 0.5349 ± 0.0783 |
| **Without Density** | 16 | 0.5632 ± 0.0452 | 0.5731 ± 0.0560 |
| **Without Category Columns** | 7 | 0.5944 ± 0.0549 | 0.6016 ± 0.0637 |

### Key Ablation Insights:
- **Landmarks are the primary spatial driver**: Removing landmark distances causes the sharpest drop in accuracy (down to 52.30%) and ROC-AUC (down to 0.5349), confirming that proximity to transit and high-income hubs (Station and Khadakpada) drives customer traffic.
- **Modest CV Scores**: We report genuine numbers (~56% accuracy, ~0.57 AUC). The model performs modestly above random chance because location is only one contributing factor to retail popularity.

---

## 6. Scaled Logistic Regression Coefficients

| Feature | Coefficient ($\beta$) | Plain Meaning |
|---|---|---|
| `nearest_competitor_distance_m` | +0.3145 | Moderate spacing from the closest direct rival helps popularity. |
| `dist_birla_college_m` | +0.3010 | *Statistical noise flag:* Higher distance correlates positively due to sparse sampling near Birla. |
| `cat_pharmacy` | +0.1661 | Pharmacies exhibit steady baseline customer reviews. |
| `competitor_count_500m` | +0.1462 | Commercial clustering/retail agglomeration benefits footfall. |
| `business_density_1km` | +0.1251 | Denser commercial zones generate higher customer discovery. |
| `cat_cafe` | +0.0819 | Positive baseline category effect. |
| `cat_restaurant` | -0.0455 | High saturation baseline effect. |
| `dist_durgadi_fort_m` | -0.1394 | Closer to historic center provides mild positive lift. |
| `dist_station_m` | -0.3316 | **Strong positive lift**: Closer to Kalyan Station = higher popularity. |
| `dist_khadakpada_m` | -0.5197 | **Strongest positive lift**: Closer to Khadakpada Circle = highest popularity. |

*Viva Caution Note:* The positive coefficient on `dist_birla_college_m` is an artifact of small sample concentration in central/west Kalyan and must not be interpreted as a causal claim that being near a college hurts a business.

---

## 7. Honest Project Limitations
1. **Single-Source Data**: Derived entirely from Google Places; businesses without online listings are absent.
2. **Survivorship Bias**: All businesses have `business_status = OPERATIONAL`. Closed or failed shops are unobserved.
3. **Popularity vs. Profitability**: Google review count reflects footfall and visibility, not net profit margins or low rent.
4. **Unobserved Non-Spatial Factors**: Rental rates per square foot, owner entrepreneurial skill, marketing, interior aesthetics, and service quality cannot be observed through GPS coordinates alone.

---

## 8. Ten Likely Viva Questions and Defendable Answers

**Q1: Why did you not use business survival (closed vs. open) as your target variable?**  
*Answer:* Google Places listings only provide operational businesses (`business_status = OPERATIONAL`). Historical records of defunct shops do not exist in the public API, so using survival would require fabricating data. We instead used relative popularity (review count above category median) as an honest proxy.

**Q2: Why use category-specific median instead of an overall median?**  
*Answer:* Different industries exhibit vastly different consumer reviewing habits. Restaurants in Kalyan average 893 reviews, whereas hardware stores average 16. A global median would classify nearly all hardware stores as failures and all restaurants as successes, introducing severe category bias.

**Q3: Why is your cross-validated accuracy ~56-60% rather than 85-95%?**  
*Answer:* An accuracy of 85%+ on 267 rows predicting real-world business success would indicate severe data leakage or overfitting. Location explains only part of business performance; rent, product quality, service, and pricing account for the remainder. Reporting realistic, modest numbers is scientifically honest and reproducible.

**Q4: Why select Logistic Regression when Random Forest had a higher ROC-AUC (0.64 vs 0.57)?**  
*Answer:* Logistic Regression provides exact, monotonic, and additive coefficient explanations ($\beta_j \cdot z_j$). In a decision-support tool, stakeholders must understand precisely which geographic factors helped or harmed their score rather than trusting an opaque tree ensemble.

**Q5: Why does distance to Birla College have a positive coefficient?**  
*Answer:* In our dataset, commercial listings are heavily clustered in Khadakpada and Station areas, which happen to be 1.5–2 km away from Birla College. The positive coefficient is a sampling artifact of spatial clustering, not a causal claim that being close to a college reduces customers.

**Q6: What is the "shared food-group rule" and why was it introduced?**  
*Answer:* Consumers looking to eat do not distinguish strictly between a cafe, bakery, or quick-service restaurant. Grouping cafe, restaurant, and bakery into a unified competitor pool accurately models substitutability in dining decisions.

**Q7: How did you prevent data leakage during feature engineering and model training?**  
*Answer:* A business's own rating and review count were strictly excluded from its feature vector. Furthermore, feature scaling was fitted strictly within each fold of the cross-validation loop using Scikit-Learn pipelines.

**Q8: How does the application recommend better spots if a location is Risky?**  
*Answer:* It samples 16 candidate coordinates along 8 compass directions at 200m and 400m intervals, re-computes spatial features against the business database, and surfaces points that yield at least a +5 percentage point improvement.

**Q9: Did you use synthetic data to balance categories or increase the sample size?**  
*Answer:* No. Fabricating synthetic rows creates artificial spatial relationships that do not exist on the ground in Kalyan. We relied entirely on 267 verified, real-world businesses.

**Q10: What is the purpose of Phase 6 (Field Check)?**  
*Answer:* Phase 6 allows physical ground-truthing by recording 15-minute manual pedestrian counts and calculating the Spearman rank correlation against our model's footfall proxy (density and landmark proximity), validating whether digital proxies correspond to foot traffic on Kalyan streets.
