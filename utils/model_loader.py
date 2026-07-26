"""
Model Loader Utility Module
Provides cached functions to load trained models and metadata.
"""
import os
import joblib
import streamlit as st

MODELS_DIR = "models"

@st.cache_resource
def load_classification_model():
    """Loads trained Random Forest classifier."""
    path = os.path.join(MODELS_DIR, "random_forest.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource
def load_clustering_model():
    """Loads trained KMeans clustering model and fitted StandardScaler."""
    kmeans_path = os.path.join(MODELS_DIR, "kmeans.joblib")
    scaler_path = os.path.join(MODELS_DIR, "kmeans_scaler.joblib")
    if not os.path.exists(kmeans_path) or not os.path.exists(scaler_path):
        return None, None
    kmeans = joblib.load(kmeans_path)
    scaler = joblib.load(scaler_path)
    return kmeans, scaler

@st.cache_resource
def load_label_encoder():
    """Loads fitted LabelEncoder for target budget_level."""
    path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_association_rules():
    """Loads mined association rules dataframe."""
    path = os.path.join(MODELS_DIR, "association_rules.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_feature_metadata():
    """Loads metadata dictionary containing feature columns, metrics, and ranges."""
    path = os.path.join(MODELS_DIR, "feature_metadata.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def verify_models_exist():
    """Checks if all required model artifacts are present."""
    required_files = [
        "random_forest.joblib",
        "kmeans.joblib",
        "kmeans_scaler.joblib",
        "label_encoder.joblib",
        "association_rules.joblib",
        "feature_metadata.joblib"
    ]
    for f in required_files:
        if not os.path.exists(os.path.join(MODELS_DIR, f)):
            return False
    return True
