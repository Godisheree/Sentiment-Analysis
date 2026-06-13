# Analisis Sentimen Bahasa Indonesia

Aplikasi **Sentiment Analysis** untuk teks berbahasa Indonesia menggunakan pendekatan Machine Learning. Proyek ini membandingkan beberapa algoritma klasifikasi dan menyediakan antarmuka web interaktif berbasis Streamlit untuk melakukan prediksi sentimen secara real-time.

## Deskripsi

Proyek ini membangun pipeline lengkap analisis sentimen mulai dari pengumpulan data, preprocessing teks bahasa Indonesia, ekstraksi fitur TF-IDF, pelatihan model, evaluasi, hingga deployment sebagai web app. Sistem mengklasifikasikan teks ke dalam tiga kategori sentimen: **Positif**, **Negatif**, dan **Netral**.

## Fitur Utama

- **Analisis Sentimen Real-time** - Input teks dan dapatkan prediksi sentimen beserta confidence score
- **Batch Analysis** - Analisis banyak teks sekaligus dalam satu kali proses
- **Perbandingan Model** - Evaluasi dan visualisasi performa 5+ algoritma ML
- **Riwayat Prediksi** - Simpan dan export hasil prediksi dalam format CSV
- **Preprocessing Bahasa Indonesia** - Normalisasi slang, stemming Sastrawi, stopword removal, dan penanganan negasi

## Model yang Dibandingkan

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Naive Bayes** | 77.86% | 0.7843 | 0.7786 | 0.7790 |
| Logistic Regression | 77.71% | 0.7776 | 0.7771 | 0.7771 |
| SVM (LinearSVC) | 76.27% | 0.7640 | 0.7627 | 0.7627 |
| Random Forest | 76.56% | 0.7661 | 0.7656 | 0.7646 |
| XGBoost | 74.96% | 0.7621 | 0.7496 | 0.7503 |

Model terbaik: **Naive Bayes** dengan akurasi **77.86%** dan CV Mean **79.75%** (+/- 0.46%).

## Pipeline Preprocessing

1. **Lowercase conversion** - Konversi seluruh teks ke huruf kecil
2. **Cleaning** - Hapus URL, mention (@), hashtag (#), dan karakter khusus
3. **Slang normalization** - Normalisasi bahasa gaul Indonesia (contoh: "gak" -> "tidak", "udah" -> "sudah")
4. **Negation handling** - Penanganan kata negasi agar tidak terbalik maknanya (contoh: "tidak bagus" -> "NOT_bagus")
5. **Stopword removal** - Hapus kata henti menggunakan Sastrawi + custom stopwords
6. **Stemming** - Kata dasar menggunakan Sastrawi Stemmer (contoh: "mencintai" -> "cinta")
7. **TF-IDF Vectorization** - Ekstraksi fitur dengan 3000 term terbobot

## Tech Stack

- **Python** 3.x
- **scikit-learn** - Machine learning models & evaluasi
- **Sastrawi** - NLP preprocessing bahasa Indonesia (stemming & stopword removal)
- **Streamlit** - Web application framework
- **XGBoost** - Gradient boosting classifier
- **NLTK** - Natural Language Toolkit
- **pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualisasi data
- **Joblib** - Model serialization

## Instalasi

```bash
# Clone repository
git clone https://github.com/Godisheree/Sentiment-Analysis.git
cd Sentiment-Analysis

# Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## Cara Menggunakan

### 1. Persiapan Data

```bash
python src/data_prep.py
```

Script ini menghasilkan dataset sentimen sintetis berbahasa Indonesia dengan 3 kelas (positif, negatif, netral).

### 2. Pelatihan Model

```bash
python src/train_improved.py
```

Pipeline ini melakukan preprocessing, ekstraksi fitur TF-IDF, training 5+ model, hyperparameter tuning dengan GridSearchCV, dan evaluasi menggunakan 5-fold cross-validation. Hasil model terbaik dan vectorizer disimpan ke folder `models/`.

### 3. Analisis Error (Opsional)

```bash
python src/error_analysis.py
```

Menganalisis kesalahan prediksi model untuk identifikasi pola kelemahan.

### 4. Model Lanjutan (Opsional)

```bash
python src/train_advanced.py
```

Melatih model tambahan seperti Neural Network (MLPClassifier).

### 5. Jalankan Web App

```bash
streamlit run src/app_improved.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

## Struktur Proyek

```
Sentiment-Analysis/
├── data/
│   ├── cleaned_data.csv            # Dataset hasil preprocessing
│   ├── cleaned_improved.csv        # Dataset improved
│   ├── combined_sentiment.csv      # Dataset gabungan
│   ├── raw_sentiment.csv           # Dataset mentah
│   ├── model_comparison.csv        # Hasil perbandingan model
│   ├── model_comparison_bar.png    # Visualisasi perbandingan
│   ├── model_comparison_cm.png     # Confusion matrix
│   ├── learning_curve.png          # Kurva pembelajaran
│   └── *.png                       # Visualisasi EDA lainnya
├── models/
│   ├── best_model.pkl              # Model terbaik (Naive Bayes)
│   ├── tfidf_vectorizer_improved.pkl  # TF-IDF vectorizer
│   ├── stemmer.pkl                 # Sastrawi Stemmer
│   ├── stopword_remover.pkl        # Sastrawi Stopword Remover
│   ├── model_metadata.json         # Metadata model
│   └── *.pkl                       # Model lainnya
├── src/
│   ├── app_improved.py             # Streamlit web app (utama)
│   ├── app.py                      # Streamlit web app (versi awal)
│   ├── data_prep.py                # Persiapan &生成an data
│   ├── train_improved.py           # Pipeline pelatihan model
│   ├── train.py                    # Training pipeline awal
│   ├── train_advanced.py           # Model lanjutan (NN)
│   ├── preprocess.py               # Fungsi preprocessing
│   ├── predict.py                  # Fungsi prediksi
│   ├── eda.py                      # Exploratory Data Analysis
│   ├── error_analysis.py           # Analisis error
│   ├── merge_real_data.py          # Merge data real
│   └── deploy.py                   # Script deployment
├── requirements.txt
└── README.md
```

## Dataset

Dataset yang digunakan adalah data sentimen sintetis berbahasa Indonesia dengan total **3.451 sampel** yang terdistribusi ke dalam 3 kelas:

- **Positif** - Ulasan positif (puas, bagus, recommended, dll.)
- **Negatif** - Ulasan negatif (kecewa, rusak, buruk, dll.)
- **Netral** - Ulasan netral (biasa, standar, cukup, dll.)

Data dihasilkan menggunakan template-based generation dengan variasi kalimat dan kosakata bahasa Indonesia sehari-hari termasuk slang dan bahasa informal.

## Kontribusi

Kontribusi sangat diterima! Silakan fork repository ini, buat branch baru, dan submit pull request.

## Lisensi

Proyek ini dibuat untuk tujuan pembelajaran dan penelitian.

## Kontak

- **GitHub:** [Godisheree](https://github.com/Godisheree)
