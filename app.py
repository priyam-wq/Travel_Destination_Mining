"""
Travel Destination Mining - Streamlit Web Application
Production-ready application implementing Random Forest Classification, K-Means Clustering,
and Apriori Association Rule Mining for Travel Destination Recommendation & Insights.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Travel Destination Mining AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import local utilities
from utils.preprocessing import parse_avg_temp_monthly, parse_list_like
from utils.model_loader import (
    load_classification_model,
    load_clustering_model,
    load_label_encoder,
    load_association_rules,
    load_feature_metadata,
    verify_models_exist
)
from utils.predictor import (
    validate_inputs,
    prepare_prediction_dataframe,
    predict_budget_level,
    predict_cluster,
    find_matching_association_rules
)
from utils.ui_helpers import (
    inject_custom_css,
    render_radar_chart,
    render_probability_chart,
    render_feature_importance_chart,
    render_confusion_matrix_chart
)

# Inject modern CSS
inject_custom_css()

# Auto-train models if missing
if not verify_models_exist():
    with st.spinner("📦 Initializing trained models from dataset... Please wait a moment."):
        import train
        train.train_and_save_models()
    st.rerun()

# Load models and metadata
clf = load_classification_model()
kmeans, scaler = load_clustering_model()
le_target = load_label_encoder()
rules_df = load_association_rules()
metadata = load_feature_metadata()

# Check if model loading succeeded
if not all([clf, kmeans, scaler, le_target, metadata]):
    st.error("❌ Failed to load model artifacts. Please run `python train.py` to regenerate model files.")
    st.stop()

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown("""
<div class="main-banner">
    <h1>✈️ Travel Destination Mining & Recommendation AI</h1>
    <p>Discover optimal budget levels, travel personality clusters, and data-driven travel insights powered by Machine Learning & Data Mining algorithms.</p>
    <div>
        <span class="badge-pill">🌲 Random Forest Classification</span>
        <span class="badge-pill">🌀 K-Means Clustering</span>
        <span class="badge-pill">🔗 Apriori Association Rules</span>
        <span class="badge-pill">⚡ Fast Offline Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR - USER INPUT CONTROLS
# ==========================================
st.sidebar.markdown("## 🧳 Preference & Travel Inputs")
st.sidebar.markdown("Customize your destination ratings and preferences below:")

user_inputs = {}

# 1. Rating Features (Sliders 0.0 to 10.0)
st.sidebar.markdown("### ⭐ Category Ratings (0 - 10)")
rating_cols = metadata.get('rating_cols', [
    'culture','adventure','nature','beaches','nightlife',
    'cuisine','wellness','urban','seclusion'
])

for col in rating_cols:
    default_val = float(metadata['rating_medians'].get(col, 5.0))
    user_inputs[col] = st.sidebar.slider(
        f"{col.title()} Rating",
        min_value=0.0,
        max_value=10.0,
        value=default_val,
        step=0.5,
        help=f"Rate your preference for {col} on a scale from 0 (low) to 10 (high)."
    )

st.sidebar.markdown("---")

# 2. Climate & Temperature Input
st.sidebar.markdown("### 🌡️ Climate Preference")
temp_min, temp_max = metadata.get('temp_range', (-10.0, 40.0))
default_temp = float(metadata['rating_medians'].get('avg_annual_temp', 20.0))

user_inputs['avg_annual_temp'] = st.sidebar.slider(
    "Average Annual Temp (°C)",
    min_value=round(temp_min, 1),
    max_value=round(temp_max, 1),
    value=round(default_temp, 1),
    step=0.5,
    help="Target destination's average annual temperature in Celsius."
)

st.sidebar.markdown("---")

# 3. Ideal Duration Multiselect
st.sidebar.markdown("### ⏱️ Ideal Trip Duration")
duration_options = metadata.get('duration_options', ['1-3 days', '4-7 days', '1-2 weeks', '2+ weeks'])
user_inputs['ideal_durations'] = st.sidebar.multiselect(
    "Select Preferred Trip Durations",
    options=duration_options,
    default=duration_options[:2] if len(duration_options) >= 2 else duration_options,
    help="Select one or more trip length preferences."
)

st.sidebar.markdown("---")

# 4. Optional Geographic Coordinates
with st.sidebar.expander("📍 Geographic Location (Optional)", expanded=False):
    default_lat = float(metadata['rating_medians'].get('latitude', 20.0))
    default_lon = float(metadata['rating_medians'].get('longitude', 0.0))

    user_inputs['latitude'] = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=default_lat, step=0.1)
    user_inputs['longitude'] = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=default_lon, step=0.1)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Predict Budget & Mine Insights")


# ==========================================
# MAIN CONTENT AREA
# ==========================================

# Perform input validation
is_valid, error_msg = validate_inputs(user_inputs, metadata)

if not is_valid:
    st.error(f"⚠️ **Input Validation Error**: {error_msg}")
    st.stop()

if predict_btn or 'prediction_done' not in st.session_state:
    st.session_state['prediction_done'] = True

    # 1. Prepare feature sample
    df_sample = prepare_prediction_dataframe(user_inputs, metadata)

    # 2. Run Classification Prediction
    predicted_budget, confidence_score, prob_dict = predict_budget_level(clf, le_target, df_sample)

    # 3. Run Clustering Prediction
    cluster_id, cluster_name, overall_score = predict_cluster(kmeans, scaler, metadata, user_inputs)

    # 4. Find Relevant Association Rules
    matched_rules = find_matching_association_rules(rules_df, predicted_budget, user_inputs['ideal_durations'])

    # ==========================================
    # DISPLAY PREDICTION RESULTS
    # ==========================================
    st.markdown("### 🎯 Machine Learning Prediction Results")

    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-card-header">💰 Predicted Budget Level</div>
            <h2 style="color: #38bdf8; font-size: 2.2rem; margin: 0.5rem 0;">{predicted_budget}</h2>
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">Predicted target classification based on Random Forest model.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-card-header">📊 Model Confidence</div>
            <h2 style="color: #10b981; font-size: 2.2rem; margin: 0.5rem 0;">{confidence_score:.1f}%</h2>
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">Random Forest class probability score.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-card-header">🌀 Destination Cluster</div>
            <h2 style="color: #f43f5e; font-size: 2.2rem; margin: 0.5rem 0;">Cluster {cluster_id}</h2>
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">K-Means group assignment.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Detailed Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Confidence & Probabilities",
        "🌀 Cluster & Preference Profile",
        "🔗 Data Mining Association Rules",
        "ℹ️ Model Information & Performance"
    ])

    # TAB 1: Probabilities
    with tab1:
        st.markdown("#### Classification Probability Distribution")
        st.write("Below is the estimated probability distribution across all budget levels:")
        render_probability_chart(prob_dict)

    # TAB 2: Clustering
    with tab2:
        st.markdown(f"#### K-Means Cluster Assignment: **{cluster_name}**")
        st.info(f"💡 Based on your overall rating score of **{overall_score:.2f} / 10** and climate preference of **{user_inputs['avg_annual_temp']}°C**, your inputs align with **{cluster_name}**.")

        col_radar, col_summary = st.columns([1, 1])
        with col_radar:
            user_ratings_dict = {c: user_inputs[c] for c in rating_cols}
            render_radar_chart(user_ratings_dict, metadata.get('cluster_summary'), cluster_id)

        with col_summary:
            st.markdown("##### Cluster Benchmark Statistics")
            if 'cluster_summary' in metadata and not metadata['cluster_summary'].empty:
                c_df = metadata['cluster_summary'].copy()
                c_df.columns = [c.replace('_', ' ').title() for c in c_df.columns]
                st.dataframe(c_df.style.highlight_max(axis=0, color='#1e3a8a'), use_container_width=True)

    # TAB 3: Association Rules
    with tab3:
        st.markdown("#### 🔗 Association Rule Insights (Apriori Mining)")
        st.write("Frequent itemsets and association rules extracted from travel destination patterns:")

        if matched_rules is not None and not matched_rules.empty:
            for idx, r in matched_rules.reset_index().iterrows():
                ant = r.get('antecedents_str', str(r.get('antecedents', '')))
                cons = r.get('consequents_str', str(r.get('consequents', '')))
                sup = r.get('support', 0.0)
                conf = r.get('confidence', 0.0)
                lift = r.get('lift', 0.0)

                st.markdown(f"""
                <div style="background:#1e293b; border-left: 4px solid #38bdf8; padding: 1rem; border-radius: 6px; margin-bottom: 0.8rem;">
                    <strong>Rule #{idx+1}:</strong> <code style="color:#38bdf8;">{ant}</code> &nbsp; ➔ &nbsp; <code style="color:#10b981;">{cons}</code><br/>
                    <small style="color:#94a3b8;">Support: <b>{sup:.3f}</b> | Confidence: <b>{conf:.3f}</b> | Lift: <b>{lift:.3f}</b></small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No specific association rules matched current criteria.")

    # TAB 4: Model Info & Performance
    with tab4:
        st.markdown("#### 🌲 Random Forest Classifier Performance & Metadata")

        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.metric("Test Accuracy Score", f"{metadata['accuracy']*100:.2f}%")
            st.markdown("##### Confusion Matrix")
            render_confusion_matrix_chart(metadata['confusion_matrix'], metadata['classes'])

        with col_m2:
            st.markdown("##### Feature Importances")
            render_feature_importance_chart(metadata['feature_importances'])

        with st.expander("📋 Full Classification Report"):
            st.code(metadata['classification_report_str'])

        with st.expander("⚙️ Model Hyperparameters & Architecture"):
            st.json({
                "Classifier": "RandomForestClassifier",
                "n_estimators": 200,
                "random_state": 42,
                "Clustering": "KMeans",
                "n_clusters": 3,
                "preprocessing_scaler": "StandardScaler",
                "association_rule_metric": "lift",
                "min_support": 0.05,
                "min_threshold": 1.0
            })

# Success banner
st.sidebar.success("✅ Model & Pipeline Ready!")
