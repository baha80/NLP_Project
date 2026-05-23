# 🧠 Sentiment Analysis Web App — ESPRIT 2024/2025

> **NLP Project** — Analyse Automatique des Sentiments pour l'Expérience Client  
> 4ème Année Informatique | Prof. Anis Bel Hadj Hassin

---

## 📁 Project Structure

```
sentiment-analysis-app/
│
├── app.py                        # 🚀 Main Streamlit entry point
├── requirements.txt              # 📦 All dependencies
├── .env.example                  # 🔐 Environment variables template
├── README.md
│
├── data/
│   ├── sample_reviews.csv        # Sample dataset for demo
│   └── data_loader.py            # Kaggle / CSV / API loader
│
├── preprocessing/
│   ├── __init__.py
│   ├── cleaner.py                # Text cleaning pipeline
│   └── vectorizer.py             # TF-IDF + compact embeddings
│
├── models/
│   ├── __init__.py
│   ├── aspect_analyzer.py        # Aspect-based sentiment by clothing topic
│   ├── decision_engine.py        # Business recommendations from thresholds
│   ├── sentiment_vader.py        # Rule-based VADER model
│   ├── sentiment_bert.py         # DistilBERT fine-tuned model
│   ├── ner.py                    # Named Entity Recognition
│   ├── nlp_pipeline.py           # End-to-end NLP workflow
│   └── topic_model.py            # LDA topic modeling
│
├── utils/
│   ├── __init__.py
│   ├── visualizations.py         # Plotly charts for the app
│   └── metrics.py                # Evaluation metrics
│
├── pages/
│   ├── 1_Dashboard.py           # KPI overview and review table
│   ├── 2_Analyze.py             # Real-time single/batch review analysis
│   ├── 2_Aspects.py             # Aspect-based sentiment explorer
│   ├── 3_Topics.py              # LDA topic modeling results
│   ├── 4_Decisions.py           # Automatic business recommendations
│   ├── 5_Trends.py              # Monthly sentiment and aspect trends
│   └── 6_NER.py                 # Named entity exploration page
│
├── assets/
│   └── style.css                 # Custom Streamlit styles
│
└── notebooks/
    ├── 00_Scraping.ipynb         # Scraping experiments
    ├── 01_EDA.ipynb              # Exploratory Data Analysis
    ├── 02_Preprocessing.ipynb    # Preprocessing pipeline
    ├── 03_Model_Training.ipynb   # Model training & evaluation
    └── 04_Topic_Modeling.ipynb   # LDA experiments
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourname/sentiment-analysis-app
cd sentiment-analysis-app
pip install -r requirements.txt
```

### 2. Download NLTK & spaCy data
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt')"
python -m spacy download en_core_web_sm
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Deploy (Streamlit Cloud)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → Deploy ✅

---

## 📊 Features

| Feature | Status |
|---|---|
| VADER Sentiment Analysis | ✅ |
| DistilBERT Classification | ✅ |
| Aspect-Based Sentiment | ✅ |
| LDA Topic Modeling | ✅ |
| Text Cleaning + Deduplication | ✅ |
| TF-IDF Vectorization | ✅ |
| Dense Document Embeddings | ✅ |
| Named Entity Recognition | ✅ |
| Rating-Based Validation Metrics | ✅ |
| Real-time Text Analysis | ✅ |
| CSV Upload & Batch Analysis | ✅ |
| Interactive Visualizations | ✅ |
| Business KPI Dashboard | ✅ |
| Decision Recommendations | ✅ |
| Temporal Trend Analysis | ✅ |
| Export Results (CSV) | ✅ |

---

## 🛠️ Tech Stack

- **NLP**: VADER, DistilBERT (HuggingFace), spaCy, NLTK
- **Preprocessing**: cleaning, deduplication, TF-IDF, TruncatedSVD embeddings
- **Topic Modeling**: LDA (Gensim)
- **Web App**: Streamlit
- **Visualization**: Plotly
- **Deployment**: Streamlit Cloud

---

## 👥 Team
- Student 1
- Student 2
- Student 3

**Supervisor**: Prof. Anis Bel Hadj Hassin — ESPRIT 2024/2025
