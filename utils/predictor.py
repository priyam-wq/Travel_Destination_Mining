"""
Predictor Utility Module
Handles input validation, feature dataframe construction, classification, clustering, and rule matching.
"""
import pandas as pd
import numpy as np

def validate_inputs(user_inputs, metadata):
    """
    Validates user inputs from Streamlit sidebar controls.
    
    Returns:
        (bool, str): (is_valid, error_message)
    """
    if not isinstance(user_inputs, dict):
        return False, "Inputs must be a valid dictionary."

    # Validate ratings (range 0 to 10)
    for rating_key in metadata.get('rating_cols', []):
        val = user_inputs.get(rating_key)
        if val is None or not (0.0 <= val <= 10.0):
            return False, f"Rating '{rating_key}' must be between 0.0 and 10.0."

    # Validate annual temperature
    temp = user_inputs.get('avg_annual_temp')
    if temp is not None:
        temp_min, temp_max = metadata.get('temp_range', (-50.0, 60.0))
        if not (-60.0 <= temp <= 60.0):
            return False, f"Average annual temperature ({temp}°C) is outside acceptable bounds (-60°C to 60°C)."

    # Validate Latitude & Longitude
    lat = user_inputs.get('latitude')
    if lat is not None and not (-90.0 <= lat <= 90.0):
        return False, f"Latitude ({lat}) must be between -90.0 and 90.0."

    lon = user_inputs.get('longitude')
    if lon is not None and not (-180.0 <= lon <= 180.0):
        return False, f"Longitude ({lon}) must be between -180.0 and 180.0."

    return True, ""

def prepare_prediction_dataframe(user_inputs, metadata):
    """
    Constructs a 1-row pandas DataFrame formatted exactly as expected by trained models.
    """
    feature_cols = metadata['feature_cols']
    duration_cols = metadata.get('duration_cols', [])
    selected_durations = user_inputs.get('ideal_durations', [])

    row = {}

    # Fill rating columns
    for col in metadata.get('rating_cols', []):
        row[col] = float(user_inputs.get(col, metadata['rating_medians'].get(col, 5.0)))

    # Fill avg_annual_temp
    if 'avg_annual_temp' in feature_cols:
        row['avg_annual_temp'] = float(user_inputs.get('avg_annual_temp', metadata['rating_medians'].get('avg_annual_temp', 20.0)))

    # Fill duration binary columns
    for col in duration_cols:
        # col is formatted as dur_<duration_name_with_underscores>
        clean_dur_name = col.replace('dur_', '').replace('_', ' ')
        # Check if selected
        is_selected = any(sel.replace(' ', '_') == col.replace('dur_', '') or sel.strip().lower() == clean_dur_name.strip().lower() for sel in selected_durations)
        row[col] = 1 if is_selected else 0

    # Fill latitude and longitude if required
    if 'latitude' in feature_cols:
        row['latitude'] = float(user_inputs.get('latitude', metadata['rating_medians'].get('latitude', 0.0)))
    if 'longitude' in feature_cols:
        row['longitude'] = float(user_inputs.get('longitude', metadata['rating_medians'].get('longitude', 0.0)))

    df_sample = pd.DataFrame([row])

    # Ensure column ordering strictly matches training dataset
    df_sample = df_sample[feature_cols]
    return df_sample

def predict_budget_level(clf, le_target, df_sample):
    """
    Predicts budget level and computes probability breakdown.
    """
    pred_idx = clf.predict(df_sample)[0]
    predicted_label = le_target.inverse_transform([pred_idx])[0]

    probabilities = clf.predict_proba(df_sample)[0]
    classes = le_target.classes_

    prob_dict = {cls_name: float(prob) for cls_name, prob in zip(classes, probabilities)}
    confidence_score = prob_dict[predicted_label] * 100.0

    return predicted_label, confidence_score, prob_dict

def predict_cluster(kmeans, scaler, metadata, user_inputs):
    """
    Scales cluster features and assigns input to a K-Means cluster.
    """
    cluster_features = metadata['cluster_features']
    row = {}

    # Calculate overall score for user input
    rating_vals = [float(user_inputs.get(c, 5.0)) for c in metadata.get('rating_cols', [])]
    overall_score = np.mean(rating_vals) if rating_vals else 5.0
    row['overall_score'] = overall_score

    if 'avg_annual_temp' in cluster_features:
        row['avg_annual_temp'] = float(user_inputs.get('avg_annual_temp', metadata['rating_medians'].get('avg_annual_temp', 20.0)))

    for c in metadata.get('rating_cols', []):
        row[c] = float(user_inputs.get(c, 5.0))

    df_cluster_input = pd.DataFrame([row])[cluster_features]
    scaled_input = scaler.transform(df_cluster_input)
    cluster_id = int(kmeans.predict(scaled_input)[0])

    cluster_names = {
        0: "Cluster 0: Moderate All-Rounders",
        1: "Cluster 1: High-Rating Premium Destinations",
        2: "Cluster 2: Niche / Specialized Experience"
    }

    return cluster_id, cluster_names.get(cluster_id, f"Cluster {cluster_id}"), overall_score

def find_matching_association_rules(rules_df, predicted_budget, user_durations):
    """
    Filters association rules relevant to predicted budget level or chosen durations.
    """
    if rules_df is None or rules_df.empty:
        return pd.DataFrame()

    matched_rules = []
    budget_lower = str(predicted_budget).lower()

    for _, row in rules_df.iterrows():
        ant = row.get('antecedents_str', str(row.get('antecedents', ''))).lower()
        cons = row.get('consequents_str', str(row.get('consequents', ''))).lower()

        # Check relevance to predicted budget or durations
        if budget_lower in ant or budget_lower in cons or any(d.lower() in ant or d.lower() in cons for d in user_durations):
            matched_rules.append(row)

    if not matched_rules:
        # Fallback to top 5 general rules by lift
        return rules_df.head(5)

    res_df = pd.DataFrame(matched_rules).drop_duplicates().head(5)
    return res_df
