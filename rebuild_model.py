"""
Fix: Balance survived/closed labels + retrain on original 267 records only
This keeps the JS BUSINESSES array consistent with training data density.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from math import radians, cos, sin, asin, sqrt
import json

np.random.seed(42)

# ── 1. Load original 267 records only ───────────────────────────────────────
df_raw = pd.read_csv('data/kalyan_businesses.csv')
df_raw = df_raw[df_raw['name'].str.startswith('Synthetic_') == False].copy()
print(f"Original records: {len(df_raw)}")

CATEGORIES = [
    'cafe', 'restaurant', 'bakery', 'pharmacy',
    'grocery_or_supermarket', 'clothing_store', 'electronics_store',
    'hardware_store', 'salon', 'stationery'
]

FOOD_GROUP = {'cafe', 'restaurant', 'bakery'}
DURGADI    = (19.2453717, 73.1186096)

CAT_DEMAND = {
    'grocery_or_supermarket': 0.72,
    'pharmacy':               0.70,
    'restaurant':             0.65,
    'bakery':                 0.60,
    'cafe':                   0.55,
    'salon':                  0.52,
    'clothing_store':         0.50,
    'electronics_store':      0.48,
    'stationery':             0.44,
    'hardware_store':         0.42,
}

def haversine(lat1, lng1, lat2, lng2):
    R = 6371000
    φ1, φ2 = radians(lat1), radians(lat2)
    dφ = radians(lat2 - lat1)
    dλ = radians(lng2 - lng1)
    a  = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return R * 2 * asin(sqrt(a))

# ── 2. Feature engineering ───────────────────────────────────────────────────
rows = df_raw.to_dict('records')
features = []

for i, r in enumerate(rows):
    lat, lng, cat = r['lat'], r['lng'], r['category_searched']

    density = sum(
        1 for j, o in enumerate(rows)
        if i != j and haversine(lat, lng, o['lat'], o['lng']) <= 1000
    )

    comp_dist = []
    for j, o in enumerate(rows):
        if i == j: continue
        d = haversine(lat, lng, o['lat'], o['lng'])
        same = (o['category_searched'] == cat) or \
               (cat in FOOD_GROUP and o['category_searched'] in FOOD_GROUP)
        if same:
            comp_dist.append(d)

    comp_500       = sum(1 for d in comp_dist if d <= 500)
    nearest_comp   = min(comp_dist) if comp_dist else 1500.0
    dist_durgadi   = haversine(lat, lng, DURGADI[0], DURGADI[1])
    rating         = r.get('rating', 3.5) or 3.5
    reviews        = min(r.get('user_ratings_total', 0) or 0, 5000)

    features.append({
        'lat': lat, 'lng': lng, 'category': cat,
        'competitor_count_500m':     comp_500,
        'nearest_competitor_dist_m': round(nearest_comp, 1),
        'business_density_1km':      density,
        'dist_durgadi_fort_m':       round(dist_durgadi, 1),
        'avg_rating':                rating,
        'review_count':              reviews,
        **{f'cat_{c}': int(cat == c) for c in CATEGORIES}
    })
    if (i+1) % 50 == 0:
        print(f"  Features: {i+1}/{len(rows)}")

df_feat = pd.DataFrame(features)
print("Feature engineering done.")

# ── 3. Create balanced survival labels ──────────────────────────────────────
def survival_prob(row):
    density_score = min(row['business_density_1km'] / 40.0, 1.0)
    comp_penalty  = max(0, 1 - row['competitor_count_500m'] / 6.0)
    spacing_score = min(row['nearest_competitor_dist_m'] / 350.0, 1.0)
    cat_score     = CAT_DEMAND.get(row['category'], 0.5)
    rating_score  = (row['avg_rating'] - 1.0) / 4.0
    review_score  = min(row['review_count'] / 300.0, 1.0)

    p = (
        0.30 * density_score  +
        0.22 * comp_penalty   +
        0.18 * spacing_score  +
        0.18 * cat_score      +
        0.08 * rating_score   +
        0.04 * review_score
    )
    noise = np.random.normal(0, 0.09)
    return np.clip(p + noise, 0.02, 0.98)

probs  = df_feat.apply(survival_prob, axis=1)

# Use median as threshold to get ~50/50 balance
threshold = probs.median()
labels = (probs >= threshold).astype(int)
df_feat['survived'] = labels
print(f"Label distribution: {labels.value_counts().to_dict()}")
print(f"Threshold used: {threshold:.4f}")

# ── 4. Train balanced model ──────────────────────────────────────────────────
FEATURE_COLS = [
    'competitor_count_500m', 'nearest_competitor_dist_m',
    'business_density_1km', 'dist_durgadi_fort_m',
    'avg_rating', 'review_count',
] + [f'cat_{c}' for c in CATEGORIES]

X = df_feat[FEATURE_COLS].values
y = df_feat['survived'].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
cv    = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
auc   = cross_val_score(model, X_sc, y, cv=cv, scoring='roc_auc')
print(f"\nROC-AUC: {auc.mean():.4f} ± {auc.std():.4f}")
model.fit(X_sc, y)
print(f"Intercept: {model.intercept_[0]:.4f}")

# ── 5. Output weights ────────────────────────────────────────────────────────
coef_dict   = {col: round(float(c), 6) for col, c in zip(FEATURE_COLS, model.coef_[0])}
coef_sorted = dict(sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True))
scaler_dict = {}
for col, mean, std in zip(FEATURE_COLS, scaler.mean_, scaler.scale_):
    scaler_dict[col] = {'mean': round(float(mean), 4), 'std': round(float(std), 4)}

result = {
    'intercept':    round(float(model.intercept_[0]), 4),
    'scaler':       scaler_dict,
    'coefficients': coef_sorted,
    'roc_auc':      round(float(auc.mean()), 4),
    'n_records':    len(df_feat)
}

print("\n=== intercept ===")
print(result['intercept'])
print("\n=== scaler ===")
for k, v in result['scaler'].items():
    print(f"  {k}: mean={v['mean']}, std={v['std']}")
print("\n=== coefficients ===")
for k, v in result['coefficients'].items():
    print(f"  {k}: {v}")

with open('data/new_model_weights.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved. ROC-AUC={result['roc_auc']} | n={result['n_records']}")
