"""
app.py
Final-Year Project: Location Intelligence System for Kalyan (Maharashtra)
A clean, student-friendly, all-in-one Streamlit application.
Everything is kept in this single file so it is easy to read, modify, and explain to teachers.
"""

import math
import requests
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="Kalyan Location Intelligence",
    page_icon="📍",
    layout="wide"
)

# Coordinates of the 4 main landmarks in Kalyan
LANDMARKS = {
    "dist_station_m": (19.2403, 73.1305),             # Kalyan Railway Station
    "dist_birla_college_m": (19.2479275, 73.1471217),   # B.K. Birla College
    "dist_khadakpada_m": (19.2532055, 73.1370898),      # Khadakpada Circle
    "dist_durgadi_fort_m": (19.2453717, 73.1186096)    # Durgadi Fort
}

# Cafe, restaurant, and bakery compete together for hungry customers
FOOD_GROUP = {"cafe", "restaurant", "bakery"}

CATEGORY_LABELS = {
    "bakery": "Bakery",
    "cafe": "Cafe",
    "clothing_store": "Clothing Store",
    "electronics_store": "Electronics Store",
    "grocery_or_supermarket": "Grocery / Supermarket",
    "hardware_store": "Hardware Store",
    "pharmacy": "Pharmacy",
    "restaurant": "Restaurant",
    "salon": "Salon",
    "stationery": "Stationery Store"
}

ALL_CATEGORIES = sorted(list(CATEGORY_LABELS.keys()))

FEATURE_LABELS = {
    "dist_khadakpada_m": "Closeness to Khadakpada Circle",
    "dist_station_m": "Closeness to Kalyan Railway Station",
    "dist_birla_college_m": "Distance from B.K. Birla College",
    "dist_durgadi_fort_m": "Closeness to Durgadi Fort area",
    "nearest_competitor_distance_m": "Spacing from nearest competitor",
    "competitor_count_500m": "Competitor clustering within 500m",
    "business_density_1km": "Commercial activity density within 1km",
    "cat_cafe": "Cafe baseline demand",
    "cat_restaurant": "Restaurant baseline demand",
    "cat_bakery": "Bakery baseline demand",
    "cat_pharmacy": "Pharmacy baseline demand",
    "cat_grocery_or_supermarket": "Grocery baseline demand",
    "cat_clothing_store": "Clothing store baseline demand",
    "cat_electronics_store": "Electronics store baseline demand",
    "cat_hardware_store": "Hardware store baseline demand",
    "cat_salon": "Salon baseline demand",
    "cat_stationery": "Stationery baseline demand"
}

# ==============================================================================
# 2. HELPER FUNCTIONS (SIMPLE MATH & SPATIAL CALCULATIONS)
# ==============================================================================

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Standard Haversine formula to find distance between two GPS points in metres.
    Easy to explain in viva: converts degrees to radians, uses Earth radius = 6371km.
    """
    earth_radius = 6371000.0  # metres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius * c

def check_is_competitor(cat1, cat2):
    """
    Returns True if two shops compete.
    Cafes, restaurants, and bakeries share the food group. Other shops only compete with same type.
    """
    if cat1 in FOOD_GROUP and cat2 in FOOD_GROUP:
        return True
    return cat1 == cat2

def calculate_point_features(lat, lng, category, businesses_df):
    """
    Computes all distance, competitor, and density numbers for any chosen point.
    """
    features = {}

    # Distance to the 4 landmarks
    for name, (l_lat, l_lng) in LANDMARKS.items():
        features[name] = round(calculate_distance(lat, lng, l_lat, l_lng), 2)

    # Competitor metrics and commercial density
    competitor_distances = []
    density_1km = 0

    for _, row in businesses_df.iterrows():
        b_lat = row['lat']
        b_lng = row['lng']
        b_cat = row['category_searched']

        dist = calculate_distance(lat, lng, b_lat, b_lng)
        if dist < 1.0:
            continue  # ignore the exact spot itself

        if dist <= 1000.0:
            density_1km += 1

        if check_is_competitor(category, b_cat):
            competitor_distances.append(dist)

    # Competitors within 500m
    features["competitor_count_500m"] = sum(1 for d in competitor_distances if d <= 500.0)

    # Distance to nearest competitor
    if competitor_distances:
        features["nearest_competitor_distance_m"] = round(min(competitor_distances), 2)
    else:
        features["nearest_competitor_distance_m"] = 5000.0

    # Total business density within 1km
    features["business_density_1km"] = density_1km

    # One-hot encoding for the selected category
    for cat in ALL_CATEGORIES:
        features[f"cat_{cat}"] = 1 if cat == category else 0

    return features

# ==============================================================================
# 3. MODEL TRAINING ON STARTUP (CACHED FOR SPEED)
# ==============================================================================

@st.cache_resource
def prepare_system():
    """
    Loads Kalyan businesses data and trains Logistic Regression in 0.1s.
    Cached so it only runs once when the app opens.
    No complicated pickle files needed!
    """
    # 1. Load data
    df = pd.read_csv("data/kalyan_businesses.csv")

    # 2. Define success label: 1 if reviews > category median, else 0
    cat_medians = df.groupby('category_searched')['user_ratings_total'].transform('median')
    df['success_label'] = (df['user_ratings_total'] > cat_medians).astype(int)

    # 3. Compute spatial features for all 267 businesses
    feature_rows = []
    for idx, row in df.iterrows():
        subset = df.drop(index=idx)  # exclude self
        f = calculate_point_features(row['lat'], row['lng'], row['category_searched'], subset)
        feature_rows.append(f)

    feats_df = pd.DataFrame(feature_rows)
    feature_cols = list(feats_df.columns)

    X = feats_df.values
    y = df['success_label'].values

    # 4. Standardize features and fit Logistic Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    return model, scaler, feature_cols, df

# ==============================================================================
# 4. ADDRESS SEARCH & PREDICTION HELPERS
# ==============================================================================

def search_address_nominatim(query_text):
    """
    Free geocoding using OpenStreetMap Nominatim.
    Appends Kalyan, Maharashtra if missing.
    """
    query = query_text.strip()
    if "kalyan" not in query.lower():
        query += ", Kalyan, Maharashtra"

    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "KalyanStudentCapstoneProject/1.0"}
    params = {"q": query, "format": "json", "limit": 1, "countrycodes": "in"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", query)
    except Exception:
        pass
    return None, None, None

def evaluate_spot(lat, lng, category, model, scaler, feature_cols, df):
    """
    Calculates features, scales them, runs model.predict_proba,
    and calculates each feature's contribution (coef * scaled_value).
    """
    feats = calculate_point_features(lat, lng, category, df)
    row_values = np.array([[feats[col] for col in feature_cols]])
    scaled_values = scaler.transform(row_values)[0]

    # Predict viability probability
    prob = model.predict_proba([scaled_values])[0, 1]

    # Calculate contribution for each feature: beta * z
    coefs = model.coef_[0]
    influences = []
    for col, coef, z in zip(feature_cols, coefs, scaled_values):
        impact = coef * z
        influences.append({
            "feature": col,
            "impact": impact,
            "abs_impact": abs(impact),
            "label": FEATURE_LABELS.get(col, col)
        })

    # Sort by strongest impact
    influences_df = pd.DataFrame(influences).sort_values(by="abs_impact", ascending=False)
    return prob, feats, influences_df

def recommend_better_spots(lat, lng, category, current_score, model, scaler, feature_cols, df):
    """
    If a spot is Risky, test 8 compass directions at 200m and 400m.
    Returns up to 3 spots scoring at least 5% higher.
    """
    directions = [
        ("North", 0), ("North-East", 45), ("East", 90), ("South-East", 135),
        ("South", 180), ("South-West", 225), ("West", 270), ("North-West", 315)
    ]
    distances = [200, 400]
    better_spots = []

    for name, deg in directions:
        rad = math.radians(deg)
        for d in distances:
            d_lat = (d * math.cos(rad)) / 111320.0
            d_lng = (d * math.sin(rad)) / (111320.0 * math.cos(math.radians(lat)))
            cand_lat = lat + d_lat
            cand_lng = lng + d_lng

            p, _, _ = evaluate_spot(cand_lat, cand_lng, category, model, scaler, feature_cols, df)
            cand_score = int(round(p * 100))

            if cand_score >= current_score + 5:
                better_spots.append({
                    "Move": f"{d}m {name}",
                    "New Score": f"{cand_score}%",
                    "Score_val": cand_score,
                    "Lat": round(cand_lat, 5),
                    "Lng": round(cand_lng, 5)
                })

    if better_spots:
        b_df = pd.DataFrame(better_spots).sort_values(by="Score_val", ascending=False).head(3)
        return b_df[["Move", "New Score", "Lat", "Lng"]]
    return pd.DataFrame()

# ==============================================================================
# 5. USER INTERFACE
# ==============================================================================

def main():
    model, scaler, feature_cols, df = prepare_system()

    st.title("📍 Kalyan Location Intelligence")
    st.markdown("Decision-support system for commercial site selection in Kalyan, Maharashtra.")

    # Default location: Khadakpada Circle
    if "lat" not in st.session_state:
        st.session_state["lat"] = 19.2532
        st.session_state["lng"] = 73.1371
    if "loc_name" not in st.session_state:
        st.session_state["loc_name"] = "Khadakpada Circle, Kalyan West"

    # Sidebar
    with st.sidebar:
        st.header("Search & Options")

        # 1. Address Search
        search_input = st.text_input("Search address / landmark in Kalyan:", value="")
        if st.button("Search Location"):
            if search_input.strip():
                f_lat, f_lng, f_name = search_address_nominatim(search_input)
                if f_lat and f_lng:
                    st.session_state["lat"] = f_lat
                    st.session_state["lng"] = f_lng
                    st.session_state["loc_name"] = f_name
                    st.success("Location found!")
                else:
                    st.error("Address not found in Kalyan. Try a landmark like 'Khadakpada Circle'.")

        # 2. Business Category Dropdown
        selected_cat = st.selectbox(
            "Select Business Type:",
            options=ALL_CATEGORIES,
            index=ALL_CATEGORIES.index("cafe"),
            format_func=lambda c: CATEGORY_LABELS[c]
        )

        # Toggle density heatmap
        enable_heatmap = st.checkbox("Show Footfall/Density Heatmap", value=False)

    # Main area split into Map and Viability Analysis
    col_map, col_details = st.columns([1.1, 1.0])

    cur_lat = st.session_state["lat"]
    cur_lng = st.session_state["lng"]

    with col_map:
        st.subheader("Interactive Kalyan Map")
        st.caption("Click any spot on the map to evaluate that location.")

        # Create Folium Map with Esri World Street Map and dark mode filter
        folium_map = folium.Map(
            location=[cur_lat, cur_lng],
            zoom_start=16,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Esri"
        )

        # Invert tile CSS for dark map mode (no paid CARTO keys required)
        dark_filter = """
        <style>
        .leaflet-tile-pane {
            filter: invert(1) hue-rotate(180deg) brightness(0.85) contrast(0.95) saturate(0.6);
        }
        </style>
        """
        folium_map.get_root().html.add_child(folium.Element(dark_filter))

        # Add Heatmap if checked
        if enable_heatmap:
            heat_points = [[r["lat"], r["lng"]] for _, r in df.iterrows()]
            HeatMap(heat_points, radius=15, blur=18).add_to(folium_map)

        # Plot nearby competitors on the map in red
        for _, shop in df.iterrows():
            d = calculate_distance(cur_lat, cur_lng, shop["lat"], shop["lng"])
            if d <= 1200 and check_is_competitor(selected_cat, shop["category_searched"]):
                folium.CircleMarker(
                    location=[shop["lat"], shop["lng"]],
                    radius=5,
                    color="#FF4B4B",
                    fill=True,
                    fill_color="#FF4B4B",
                    fill_opacity=0.8,
                    tooltip=f"{shop['name']} ({shop['category_searched']}) - {int(d)}m away"
                ).add_to(folium_map)

        # Plot chosen spot in blue
        folium.Marker(
            location=[cur_lat, cur_lng],
            tooltip="Selected Spot",
            icon=folium.Icon(color="blue", icon="star")
        ).add_to(folium_map)

        # Render map and listen for clicks
        clicked_map = st_folium(folium_map, width=580, height=520, returned_objects=["last_clicked"])

        if clicked_map and clicked_map.get("last_clicked"):
            new_lat = clicked_map["last_clicked"]["lat"]
            new_lng = clicked_map["last_clicked"]["lng"]
            if (abs(new_lat - cur_lat) > 1e-4) or (abs(new_lng - cur_lng) > 1e-4):
                st.session_state["lat"] = new_lat
                st.session_state["lng"] = new_lng
                st.session_state["loc_name"] = f"Map Pin ({new_lat:.4f}, {new_lng:.4f})"
                st.rerun()

    with col_details:
        st.subheader("Viability Analysis")

        # Evaluate selected spot
        prob, pt_feats, inf_df = evaluate_spot(
            cur_lat, cur_lng, selected_cat, model, scaler, feature_cols, df
        )
        score_pct = int(round(prob * 100))

        st.write(f"**Viability Score:** `{score_pct}%`")
        st.progress(score_pct / 100.0)

        if score_pct >= 50:
            st.success(f"✅ **Good Location** ({score_pct}% viability)")
        else:
            st.warning(f"⚠️ **Risky Location** ({score_pct}% viability)")

        # Top 4 Influencing Factors
        st.markdown("#### What influenced this score:")
        for _, factor in inf_df.head(4).iterrows():
            val = factor["impact"]
            icon = "⬆️" if val >= 0 else "⬇️"
            impact_desc = "Helped score" if val >= 0 else "Lowered score"
            st.write(f"{icon} **{factor['label']}** ({impact_desc}: `{val:+.2f}`)")

        # Local stats
        st.markdown("#### Location Details:")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"• Competitors in 500m: **{pt_feats['competitor_count_500m']}**")
            st.write(f"• Nearest rival: **{pt_feats['nearest_competitor_distance_m']:.0f} m**")
            st.write(f"• Commercial density (1km): **{pt_feats['business_density_1km']}**")
        with c2:
            st.write(f"• To Kalyan Station: **{pt_feats['dist_station_m']:.0f} m**")
            st.write(f"• To Khadakpada Circle: **{pt_feats['dist_khadakpada_m']:.0f} m**")
            st.write(f"• To Birla College: **{pt_feats['dist_birla_college_m']:.0f} m**")
            st.write(f"• To Durgadi Fort: **{pt_feats['dist_durgadi_fort_m']:.0f} m**")

        # Recommendations for Risky Spot
        if score_pct < 50:
            st.markdown("---")
            st.markdown("#### Recommendations for this Risky Spot:")

            better_spots = recommend_better_spots(
                cur_lat, cur_lng, selected_cat, score_pct, model, scaler, feature_cols, df
            )
            if not better_spots.empty:
                st.write(f"**Better nearby spots for a {CATEGORY_LABELS[selected_cat]}:**")
                for _, b in better_spots.iterrows():
                    st.write(f"👉 Move **{b['Move']}** → Viability jumps to **{b['New Score']}**")

            # Better business types at this spot
            viable_types = []
            for cat in ALL_CATEGORIES:
                p_cat, _, _ = evaluate_spot(cur_lat, cur_lng, cat, model, scaler, feature_cols, df)
                sc = int(round(p_cat * 100))
                if sc >= 50:
                    viable_types.append((CATEGORY_LABELS[cat], sc))
            if viable_types:
                viable_types.sort(key=lambda x: x[1], reverse=True)
                st.write(f"**Alternative businesses that would score Good here:**")
                for name, sc in viable_types[:3]:
                    st.write(f"💼 **{name}** ({sc}% viability)")

    # Compare all business categories table
    st.markdown("---")
    st.subheader("Compare All Business Types at this Location")
    comparison = []
    for cat in ALL_CATEGORIES:
        p_c, _, _ = evaluate_spot(cur_lat, cur_lng, cat, model, scaler, feature_cols, df)
        sc = int(round(p_c * 100))
        comparison.append({
            "Business Category": CATEGORY_LABELS[cat],
            "Viability Score": f"{sc}%",
            "Assessment": "Good" if sc >= 50 else "Risky",
            "_sort": sc
        })
    comp_df = pd.DataFrame(comparison).sort_values(by="_sort", ascending=False).drop(columns=["_sort"])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Disclaimer
    st.caption("Score reflects location conditions, not a guarantee of success.")

if __name__ == "__main__":
    main()
