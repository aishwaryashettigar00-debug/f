# 🔬 FeedbackLens — AI Customer Feedback Clustering Engine

> Upload any customer feedback dataset → AI cleans it → Clusters by theme → Feature recommendations & priority matrix for product managers

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## What It Does

FeedbackLens is a complete, end-to-end AI pipeline that:

1. **Loads** any CSV or Excel feedback file (auto-detects column names)
2. **Cleans** text — removes URLs, symbols, stop-words, duplicates
3. **Vectorises** using TF-IDF (500 features, bigrams)
4. **Clusters** with K-Means + auto Silhouette optimisation
5. **Visualises** clusters in 2D with PCA scatter plot
6. **Recommends** feature improvements per cluster
7. **Exports** a full Excel report with 3 sheets

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Streamlit** | Web app framework |
| **scikit-learn** | TF-IDF, K-Means, PCA, Silhouette |
| **pandas** | Data manipulation |
| **matplotlib** | Charts & visualisations |
| **openpyxl** | Excel export |

---

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/feedbacklens.git
cd feedbacklens
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy on Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub → select this repo → `app.py`
4. Click **Deploy** — live in ~2 minutes

---

## Dataset Format

Works with **any** CSV or Excel file. Minimum: one text column with reviews.

| Column | Required | Notes |
|--------|----------|-------|
| Feedback Text / Review / Comment | ✅ Yes | Any name — auto-detected |
| Rating / Score / Stars | Optional | 1–5 scale → sentiment labels |
| Source / Platform | Optional | Amazon, Google Play, etc. |
| Brand / Company | Optional | For filtering |
| Product / Item | Optional | For grouping |
| Category | Optional | Electronics, Beauty, etc. |
| Date | Optional | Review date |

---

## AI Business Project — Academic Context

Built for an **AI in Business** course to demonstrate:
- Real-world NLP pipeline on 547 actual customer reviews
- Unsupervised ML (K-Means clustering) with auto K-detection
- Business-ready output: priority matrix, feature recommendations
- Live interactive Streamlit application

---

*FeedbackLens v2.0 · Built with Python, Scikit-Learn & Streamlit*
