# ✈️ Travel Destination Mining & Recommendation Streamlit Web Application

A complete, production-ready Streamlit web application for travel destination recommendation and data mining. This project transforms raw travel city rating and climate data into actionable insights using supervised classification, unsupervised clustering, and market basket association rule mining.

---

## 🌟 Key Features

- **Machine Learning Classification**: Predicts destination `budget_level` using a Random Forest Classifier trained on ratings, climate, and duration features.
- **Destination Clustering**: Groups destinations into 3 distinct K-Means clusters based on overall scores, temperature, and category ratings.
- **Association Rule Mining**: Discovers frequent travel itemsets and pattern rules using Apriori algorithm and Lift metrics.
- **Fast Startup & Instant Inference**: Models are serialized into `.joblib` artifacts using a standalone training script (`train.py`), preventing expensive retraining on application launch.
- **Modern UI / UX**: Built with custom glassmorphism CSS, interactive radar charts, probability distribution graphs, and responsive sidebars with real-time validation.

---

## 📁 Project Structure

```
c:/CvProject/
├── app.py                      # Main Streamlit Web Application
├── train.py                    # Standalone model training & artifact export script
├── Worldwide Travel Cities...  # Primary Dataset CSV
├── models/                     # Serialized ML model artifacts (.joblib)
│   ├── random_forest.joblib
│   ├── kmeans.joblib
│   ├── kmeans_scaler.joblib
│   ├── label_encoder.joblib
│   ├── association_rules.joblib
│   └── feature_metadata.joblib
├── utils/                      # Modular python utilities
│   ├── __init__.py
│   ├── preprocessing.py       # Data parsing & cleaning (exact notebook logic)
│   ├── model_loader.py        # Streamlit resource caching (@st.cache_resource)
│   ├── predictor.py           # Feature construction, inference, cluster & rule matching
│   └── ui_helpers.py          # Custom CSS, radar plots & metric charts
├── assets/                     # Stylesheets and visual assets
│   └── style.css
├── requirements.txt            # Project dependencies
├── README.md                   # Application documentation
└── .gitignore                  # Git ignore rules
```

---

## 🚀 Commands Required to Run Locally

### 1. Prerequisite & Virtual Environment Setup
Open a terminal in the project directory:

```bash
# Option A: Windows PowerShell
py -3.10 -m venv .venv
.\.venv\Scripts\activate

# Option B: macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate / Train Saved Model Artifacts
Before running the Streamlit app for the first time, execute the offline training script to generate all `.joblib` files in `models/`:

```bash
python train.py
```

### 4. Launch Streamlit Application

```bash
# Recommended for Windows & cross-platform execution (works even if Scripts is not on PATH):
python -m streamlit run app.py

# Or if virtualenv is activated:
streamlit run app.py
```


The application will launch in your browser at `http://localhost:8501`.

---

## ☁️ Commands & Steps Required to Deploy on Streamlit Community Cloud

### Step 1: Initialize Git & Push to GitHub
Make sure your workspace is committed and pushed to a public or private GitHub repository:

```bash
git init
git add .
git commit -m "Initial commit of Travel Destination Mining Streamlit App"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/travel-destination-mining.git
git push -u origin main
```

> **Note**: Ensure the `models/` directory (with `.joblib` files) is committed so Streamlit Community Cloud loads pre-trained models instantly without needing to train on startup.

### Step 2: Deploy on Streamlit Community Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io/).
2. Click **"New app"**.
3. Select your repository: `YOUR_USERNAME/travel-destination-mining`.
4. Set Branch: `main`.
5. Set Main file path: `app.py`.
6. Click **"Deploy!"**.

Streamlit Community Cloud will automatically install dependencies from `requirements.txt` and run `app.py`.

---

## 🔬 Machine Learning Pipeline Architecture

| Task | Algorithm | Features Used | Output / Metrics |
| :--- | :--- | :--- | :--- |
| **Classification** | Random Forest Classifier ($N=200$) | Category ratings, `avg_annual_temp`, `dur_*` binary features, `lat`, `lon` | `budget_level` prediction & confidence score (%) |
| **Clustering** | K-Means Clustering ($K=3$) | `overall_score`, `avg_annual_temp`, Category ratings scaled via `StandardScaler` | Cluster ID (0, 1, 2) & preference radar chart |
| **Association Rules** | Apriori + Association Rules | `budget_level`, `country`, `dur_*` binary features | Frequent itemsets, Support, Confidence, Lift |

---

## 🛡️ Input Validation & Error Handling

- **Rating Inputs**: Validated to be within `[0.0, 10.0]`.
- **Temperature Input**: Validated to be within dataset bounds (`[-60°C, 60°C]`).
- **Geographic Input**: Enforces valid range for Latitude (`[-90, 90]`) and Longitude (`[-180, 180]`).
- **Automatic Model Recovery**: If `.joblib` files are missing, `app.py` automatically detects this and triggers `train.train_and_save_models()` to build them.
