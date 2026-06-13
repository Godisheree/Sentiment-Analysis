# Analisis Sentimen Bahasa Indonesia

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/Godisheree/Sentiment-Analysis/actions/workflows/ci.yml/badge.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

Aplikasi **Sentiment Analysis** untuk teks berbahasa Indonesia menggunakan pendekatan Machine Learning dan Deep Learning. Proyek ini membandingkan beberapa algoritma klasifikasi (Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost, Neural Network, BERT) dan menyediakan antarmuka web interaktif berbasis Streamlit untuk prediksi sentimen secara real-time.

## Quick Start

```bash
git clone https://github.com/Godisheree/Sentiment-Analysis.git
cd Sentiment-Analysis
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
streamlit run src/app_improved.py
```

App will open at `http://localhost:8501`.

## Features

- **Real-time Sentiment Analysis** - Input text and get sentiment prediction with confidence score
- **Batch Analysis** - Analyze multiple texts in one go
- **Model Comparison Dashboard** - Compare 5+ ML algorithms side by side
- **Prediction History** - Track and export predictions as CSV
- **Indonesian NLP Preprocessing** - Slang normalization, Sastrawi stemming, stopword removal, negation handling
- **BERT Deep Learning** - Fine-tune IndoBERT for state-of-the-art accuracy
- **Docker Support** - One-command deployment
- **CI/CD Pipeline** - Automated testing on every push

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Naive Bayes** | 77.86% | 0.7843 | 0.7786 | 0.7790 |
| Logistic Regression | 77.71% | 0.7776 | 0.7771 | 0.7771 |
| SVM (LinearSVC) | 76.27% | 0.7640 | 0.7627 | 0.7627 |
| Random Forest | 76.56% | 0.7661 | 0.7656 | 0.7646 |
| XGBoost | 74.96% | 0.7621 | 0.7496 | 0.7503 |

Best model: **Naive Bayes** with **77.86% accuracy** and CV Mean **79.75%** (+/- 0.46%).

### Performance Benchmarks

| Metric | Value |
|--------|-------|
| Best Accuracy | 77.86% |
| Best F1-Score | 0.7790 |
| CV Mean (5-Fold) | 79.75% |
| CV Std | 0.46% |
| Training Samples | 3,451 |
| TF-IDF Features | 3,000 |
| Inference Time | ~5-10ms per text |

## Preprocessing Pipeline

1. **Lowercase conversion** - Convert all text to lowercase
2. **Cleaning** - Remove URLs, mentions (@), hashtags (#), special characters
3. **Slang normalization** - Normalize Indonesian slang (e.g., "gak" -> "tidak", "udah" -> "sudah")
4. **Negation handling** - Prefix negated words with NOT_ (e.g., "tidak bagus" -> "NOT_bagus")
5. **Stopword removal** - Remove stopwords using Sastrawi + custom stopwords
6. **Stemming** - Extract root words using Sastrawi Stemmer (e.g., "mencintai" -> "cinta")
7. **TF-IDF Vectorization** - Extract features with 3,000 weighted terms (bigrams included)

## Tech Stack

- **Python** 3.10+
- **scikit-learn** - ML models & evaluation
- **Sastrawi** - Indonesian NLP (stemming & stopword removal)
- **Streamlit** - Web application
- **Transformers + PyTorch** - BERT deep learning model
- **XGBoost** - Gradient boosting classifier
- **pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **Docker** - Containerization
- **GitHub Actions** - CI/CD

## Installation

### Local Setup

```bash
git clone https://github.com/Godisheree/Sentiment-Analysis.git
cd Sentiment-Analysis

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Docker Setup

```bash
docker build -t sentiment-analysis .
docker run -p 8501:8501 sentiment-analysis

# Or with docker-compose
docker-compose up
```

### Installation Troubleshooting

| Issue | Solution |
|-------|----------|
| `sastrawi` fails to install | Run `pip install --upgrade pip setuptools` first |
| `torch` installation error | Use `pip install torch --index-url https://download.pytorch.org/whl/cpu` for CPU-only |
| Port 8501 in use | Run `streamlit run src/app_improved.py --server.port 8502` |
| Model not found | Run `python src/train_improved.py` first to generate models |
| `transformers` import error | Run `pip install transformers torch` separately |

## Usage

### 1. Data Preparation

```bash
python src/data_prep.py
```

Generates a synthetic Indonesian sentiment dataset with 3 classes (positive, negative, neutral).

### 2. Model Training

```bash
python src/train_improved.py
```

Full pipeline: preprocessing, TF-IDF extraction, training 5+ models, hyperparameter tuning (GridSearchCV), and 5-fold cross-validation. Best model saved to `models/`.

### 3. Error Analysis (Optional)

```bash
python src/error_analysis.py
```

Analyze misclassified predictions to identify model weaknesses.

### 4. Advanced Models (Optional)

```bash
python src/train_advanced.py     # Neural Network (MLPClassifier)
python src/models/bert_sentiment.py  # BERT fine-tuning (needs GPU recommended)
```

### 5. Run Web App

```bash
streamlit run src/app_improved.py
```

### 6. Run Tests

```bash
pytest tests/ -v --cov=src
```

## API Usage

Use the `SentimentPredictor` class for programmatic inference:

```python
from src.inference import SentimentPredictor

predictor = SentimentPredictor()

# Single prediction
result = predictor.predict("Barang ini sangat bagus dan berkualitas!")
print(result)
# {'sentiment': 'positive', 'confidence': 0.89, 'probabilities': {...}, 'cleaned_text': '...'}

# Batch prediction
results = predictor.predict_batch([
    "Produk sangat memuaskan",
    "Pelayanan buruk sekali",
    "Biasa saja sesuai harga",
])
for r in results:
    print(f"{r['sentiment']} ({r['confidence']*100:.1f}%)")
```

## Deployment

### Streamlit Cloud (Recommended)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select: `Godisheree/Sentiment-Analysis`
5. Branch: `main`
6. Main file: `src/app_improved.py`
7. Click Deploy

### Docker

```bash
docker build -t sentiment-analysis .
docker run -d -p 8501:8501 --name sentiment-app sentiment-analysis
```

### Docker Compose

```bash
docker-compose up -d
```

### Local

```bash
streamlit run src/app_improved.py --server.port 8501 --server.address 0.0.0.0
```

## Project Structure

```
Sentiment-Analysis/
├── .github/workflows/ci.yml     # CI/CD pipeline
├── streamlit/
│   ├── config.toml              # Streamlit theme config
│   └── secrets.toml.example     # Secrets template
├── src/
│   ├── config.py                # Central configuration
│   ├── logger.py                # Logging setup
│   ├── inference.py             # Prediction pipeline
│   ├── data/
│   │   ├── loader.py            # Data loading
│   │   └── preprocessor.py      # Text preprocessing
│   ├── models/
│   │   ├── traditional.py       # NB, LR, SVM, RF, XGBoost
│   │   ├── bert_sentiment.py    # BERT deep learning
│   │   └── utils.py             # Model save/load
│   ├── evaluation/
│   │   └── metrics.py           # Metrics & visualization
│   ├── app_improved.py          # Streamlit web app (main)
│   ├── train_improved.py        # Training pipeline
│   ├── train_advanced.py        # Neural Network training
│   └── ...                      # Other scripts
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model_training.py
│   ├── test_prediction.py
│   └── test_app.py
├── data/                        # Datasets & visualizations
├── models/                      # Trained models (.pkl)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── CONTRIBUTING.md
└── README.md
```

## Dataset

Synthetic + real Indonesian sentiment data with **3,451+ samples** across 3 classes:

- **Positive** - Satisfied reviews (bagus, puas, recommended, etc.)
- **Negative** - Negative reviews (kecewa, rusak, buruk, etc.)
- **Neutral** - Neutral reviews (biasa, standar, cukup, etc.)

Data sources include template-based generation and real Indonesian tweets (telco, election, TV shows).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is created for educational and research purposes.

## Contact

- **GitHub:** [Godisheree](https://github.com/Godisheree)
