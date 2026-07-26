"""
Data Preprocessing Utility Module
Preserves exact notebook data transformation logic.
"""
import json
import ast
import pandas as pd
import numpy as np

def parse_avg_temp_monthly(val):
    """Parses JSON or stringified dict of monthly temperatures and returns average annual temp."""
    if pd.isna(val):
        return np.nan
    try:
        d = json.loads(val)
    except Exception:
        try:
            d = ast.literal_eval(val)
        except Exception:
            return np.nan
    avgs = []
    if isinstance(d, dict):
        for _, v in d.items():
            if isinstance(v, dict) and "avg" in v:
                try:
                    avgs.append(float(v["avg"]))
                except Exception:
                    pass
            else:
                try:
                    avgs.append(float(v))
                except Exception:
                    pass
    return np.mean(avgs) if avgs else np.nan

def parse_list_like(val):
    """Parses list-like string format into a python list of clean strings."""
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
        return [str(parsed)]
    except Exception:
        s2 = str(val).strip("[]")
        return [i.strip().strip('\'"') for i in s2.split(',') if i.strip()]

def preprocess_dataframe(df):
    """
    Applies exact feature engineering and preprocessing steps from the notebook.
    
    Returns:
        df_clean (pd.DataFrame): Processed dataframe
        rating_cols (list): List of rating feature column names
        sorted_durations (list): List of unique ideal duration labels
    """
    df = df.copy()

    # 1. Parse temperature
    if 'avg_temp_monthly' in df.columns:
        df['avg_annual_temp'] = df['avg_temp_monthly'].apply(parse_avg_temp_monthly)

    # 2. Parse durations and generate binary columns
    if 'ideal_durations' in df.columns:
        df['ideal_durations_list'] = df['ideal_durations'].apply(parse_list_like)
        all_durations = set(d for sub in df['ideal_durations_list'] for d in sub if d)
        sorted_durations = sorted(all_durations)
        for dur in sorted_durations:
            col_name = f"dur_{dur.replace(' ', '_')}"
            df[col_name] = df['ideal_durations_list'].apply(lambda lst: int(dur in lst))
    else:
        sorted_durations = []

    # 3. Rating columns
    rating_cols = [c for c in ['culture','adventure','nature','beaches','nightlife',
                               'cuisine','wellness','urban','seclusion'] if c in df.columns]

    # Fill median missing values
    for c in rating_cols:
        df[c] = df[c].fillna(df[c].median())

    if 'avg_annual_temp' in df.columns:
        df['avg_annual_temp'] = df['avg_annual_temp'].fillna(df['avg_annual_temp'].median())

    if 'budget_level' in df.columns:
        df['budget_level'] = df['budget_level'].fillna("Unknown")

    # 4. Overall score calculation
    df['overall_score'] = df[rating_cols].mean(axis=1) if rating_cols else 0.0

    return df, rating_cols, sorted_durations
