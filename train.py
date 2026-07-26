"""
Standalone Training Script for Travel Destination Mining Pipeline
Executes original ML workflow from notebook and exports trained model artifacts.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np

# Force UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlxtend.frequent_patterns import apriori, association_rules

from utils.preprocessing import preprocess_dataframe

def train_and_save_models():
    dataset_path = "Worldwide Travel Cities Dataset (Ratings and Climate) (1).csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    print("🚀 Loading dataset:", dataset_path)
    raw_df = pd.read_csv(dataset_path)

    # 1. Preprocessing & Feature Engineering
    print("🧹 Running preprocessing and feature engineering...")
    df, rating_cols, sorted_durations = preprocess_dataframe(raw_df)

    # Calculate median values for inference fallbacks
    rating_medians = {c: float(df[c].median()) for c in rating_cols}
    if 'avg_annual_temp' in df.columns:
        rating_medians['avg_annual_temp'] = float(df['avg_annual_temp'].median())
    if 'latitude' in df.columns:
        rating_medians['latitude'] = float(df['latitude'].median())
    if 'longitude' in df.columns:
        rating_medians['longitude'] = float(df['longitude'].median())

    # Range values for UI bounds & validation
    temp_min = float(df['avg_annual_temp'].min()) if 'avg_annual_temp' in df.columns else 0.0
    temp_max = float(df['avg_annual_temp'].max()) if 'avg_annual_temp' in df.columns else 40.0
    lat_min = float(df['latitude'].min()) if 'latitude' in df.columns else -90.0
    lat_max = float(df['latitude'].max()) if 'latitude' in df.columns else 90.0
    lon_min = float(df['longitude'].min()) if 'longitude' in df.columns else -180.0
    lon_max = float(df['longitude'].max()) if 'longitude' in df.columns else 180.0

    # 2. Clustering (K-Means)
    print("🌀 Training K-Means clustering model...")
    cluster_features = ['overall_score'] + ([ 'avg_annual_temp' ] if 'avg_annual_temp' in df.columns else []) + rating_cols
    X_cluster = df[cluster_features].copy().fillna(df[cluster_features].median())

    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_cluster_scaled)

    # Compute cluster statistics for UI display
    cluster_summary = df.groupby('cluster')[cluster_features].mean().reset_index()

    # 3. Supervised Classification (Random Forest)
    print("🌲 Training Random Forest classifier...")
    TARGET = 'budget_level'
    feature_cols = rating_cols.copy()
    if 'avg_annual_temp' in df.columns:
        feature_cols.append('avg_annual_temp')

    duration_cols = [c for c in df.columns if c.startswith('dur_')]
    feature_cols += duration_cols

    for c in ['latitude', 'longitude']:
        if c in df.columns:
            feature_cols.append(c)

    X = df[feature_cols]
    y = df[TARGET]

    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=le_target.classes_)
    cm = confusion_matrix(y_test, y_pred)

    feat_importances = dict(zip(feature_cols, clf.feature_importances_))

    print(f"✅ Classification Accuracy: {acc:.4f}")

    # 4. Association Rule Mining (Apriori)
    print("🔗 Mining association rules with Apriori...")
    basket = pd.DataFrame()
    basket['budget_level'] = df['budget_level'].astype(str)
    if 'country' in df.columns:
        basket['country'] = df['country'].astype(str)
    for c in duration_cols:
        basket[c] = df[c].astype(int)

    basket_encoded = pd.get_dummies(basket.astype(str))

    frequent_itemsets = apriori(basket_encoded, min_support=0.05, use_colnames=True)
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        if not rules.empty:
            rules_sorted = rules.sort_values(by='lift', ascending=False).head(15).reset_index(drop=True)
            # Format antecedents and consequents as lists/strings for clean rendering
            rules_sorted['antecedents_str'] = rules_sorted['antecedents'].apply(lambda x: ", ".join(list(x)))
            rules_sorted['consequents_str'] = rules_sorted['consequents'].apply(lambda x: ", ".join(list(x)))
        else:
            rules_sorted = pd.DataFrame()
    else:
        rules_sorted = pd.DataFrame()

    # 5. Export Artifacts
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(clf, os.path.join(models_dir, "random_forest.joblib"))
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "kmeans_scaler.joblib"))
    joblib.dump(le_target, os.path.join(models_dir, "label_encoder.joblib"))
    joblib.dump(rules_sorted, os.path.join(models_dir, "association_rules.joblib"))

    metadata = {
        "feature_cols": feature_cols,
        "cluster_features": cluster_features,
        "rating_cols": rating_cols,
        "duration_cols": duration_cols,
        "duration_options": sorted_durations,
        "rating_medians": rating_medians,
        "temp_range": (temp_min, temp_max),
        "lat_range": (lat_min, lat_max),
        "lon_range": (lon_min, lon_max),
        "accuracy": acc,
        "classification_report_dict": report_dict,
        "classification_report_str": report_str,
        "confusion_matrix": cm,
        "classes": list(le_target.classes_),
        "feature_importances": feat_importances,
        "cluster_summary": cluster_summary
    }

    joblib.dump(metadata, os.path.join(models_dir, "feature_metadata.joblib"))

    print("💾 All models and pipeline metadata successfully saved to models/")

if __name__ == "__main__":
    train_and_save_models()
