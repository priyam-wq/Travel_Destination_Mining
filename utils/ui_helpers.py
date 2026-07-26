"""
UI Helpers Module
Handles custom styling, interactive visualizations, and UI metrics.
"""
import os
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def inject_custom_css():
    """Injects professional custom CSS theme."""
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        # Fallback glassmorphism styling
        fallback_css = """
        <style>
        .main-header {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            margin-bottom: 2rem;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            margin-bottom: 1rem;
        }
        .badge {
            background-color: #3b82f6;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin-right: 0.5rem;
        }
        </style>
        """
        st.markdown(fallback_css, unsafe_allow_html=True)

def render_radar_chart(user_ratings, cluster_summary, assigned_cluster_id):
    """
    Renders a radar chart comparing user preference ratings with assigned cluster averages.
    """
    categories = list(user_ratings.keys())
    N = len(categories)
    if N == 0:
        return

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    user_vals = list(user_ratings.values())
    user_vals += user_vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#161b22')

    # Draw User line
    ax.plot(angles, user_vals, linewidth=2, linestyle='solid', label='Your Input Profile', color='#38bdf8')
    ax.fill(angles, user_vals, color='#38bdf8', alpha=0.25)

    # Draw Cluster average line if available
    if cluster_summary is not None and not cluster_summary.empty:
        c_row = cluster_summary[cluster_summary['cluster'] == assigned_cluster_id]
        if not c_row.empty:
            c_vals = [c_row[col].values[0] for col in categories if col in c_row.columns]
            if len(c_vals) == len(user_ratings):
                c_vals += c_vals[:1]
                ax.plot(angles, c_vals, linewidth=2, linestyle='dashed', label=f'Cluster {assigned_cluster_id} Average', color='#f43f5e')
                ax.fill(angles, c_vals, color='#f43f5e', alpha=0.15)

    plt.xticks(angles[:-1], [c.title() for c in categories], color='white', size=10)
    ax.tick_params(colors='white')
    ax.spines['polar'].set_color('#30363d')
    ax.grid(color='#30363d', linestyle='--', alpha=0.7)

    plt.title("Preference Profile vs Cluster Benchmark", color='white', size=12, pad=20, weight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
    st.pyplot(fig)
    plt.close(fig)

def render_probability_chart(prob_dict):
    """Renders a styled bar chart for class probabilities."""
    labels = list(prob_dict.keys())
    values = [v * 100 for v in prob_dict.values()]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#161b22')

    bars = ax.barh(labels, values, color='#3b82f6', edgecolor='#60a5fa', height=0.55)

    # Add text labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%",
                va='center', color='white', fontweight='bold', fontsize=10)

    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", color='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    ax.grid(axis='x', color='#30363d', linestyle='--', alpha=0.5)

    st.pyplot(fig)
    plt.close(fig)

def render_feature_importance_chart(feat_importances):
    """Renders feature importances bar chart."""
    sorted_feats = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)[:10]
    names = [x[0].replace('dur_', 'Duration: ').replace('_', ' ').title() for x in sorted_feats]
    scores = [x[1] for x in sorted_feats]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#161b22')

    ax.barh(names[::-1], scores[::-1], color='#10b981', edgecolor='#34d399', height=0.55)
    ax.set_xlabel("Importance Score", color='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    ax.grid(axis='x', color='#30363d', linestyle='--', alpha=0.5)

    plt.title("Top 10 Most Influential Features (Random Forest)", color='white', pad=15, weight='bold')
    st.pyplot(fig)
    plt.close(fig)

def render_confusion_matrix_chart(cm, classes):
    """Renders confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#161b22')

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                cbar=False, annot_kws={"size": 12, "weight": "bold"})

    ax.set_xlabel("Predicted Label", color='white', weight='bold')
    ax.set_ylabel("True Label", color='white', weight='bold')
    ax.tick_params(colors='white')
    plt.title("Confusion Matrix", color='white', pad=12, weight='bold')
    st.pyplot(fig)
    plt.close(fig)
