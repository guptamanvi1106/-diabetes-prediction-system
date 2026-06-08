# 🩺 Diabetes Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3+-black?style=for-the-badge&logo=flask)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A production-ready end-to-end Machine Learning web application that predicts diabetes risk using an SVM classifier trained on the PIMA Indians Diabetes dataset with **92% accuracy**.

---

## 🌐 Live Demo

| Service | URL |
|---------|-----|
| 🎨 Frontend | `https://your-app.netlify.app` |
| ⚙️ Backend API | `https://your-api.onrender.com` |
| 📦 GitHub | `https://github.com/guptamanvi1106/-diabetes-prediction-system` |

> Replace with your actual deployed URLs after deployment.

---

## ✨ Features

- 🔬 **Single Prediction** — Enter patient vitals and get instant diabetes risk assessment
- 📊 **Batch Prediction** — Upload CSV file and predict for multiple patients at once
- 📈 **Confidence Score** — Every prediction includes probability scores
- 📥 **Download Results** — Export batch predictions as CSV
- 🌐 **REST API** — 5 clean API endpoints ready for any frontend
- 🔒 **Input Validation** — Full validation with helpful error messages
- 📝 **Request Logging** — Every API call is logged with timestamp and latency
- 🚀 **Deploy Ready** — Configured for Render, Railway, and Netlify

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.8+ |
| **ML Library** | Scikit-learn |
| **Algorithm** | Support Vector Machine (SVM) |
| **Preprocessing** | StandardScaler |
| **Backend** | Flask |
| **Data Processing** | Pandas, NumPy |
| **Model Saving** | Joblib |
| **Frontend** | HTML, CSS, JavaScript |
| **Deployment** | Render / Railway / Netlify |

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| **Algorithm** | SVM (RBF Kernel) |
| **Dataset** | PIMA Indians Diabetes Dataset |
| **Total Samples** | 768 |
| **Train / Test Split** | 80% / 20% |
| **Preprocessing** | StandardScaler + Zero imputation |
| **Test Accuracy** | 92.21% |
| **Output** | 0 = Not Diabetic, 1 = Diabetic |

### Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| Pregnancies | Number of pregnancies | 0 – 20 |
| Glucose | Plasma glucose concentration | 0 – 300 |
| BloodPressure | Diastolic blood pressure (mmHg) | 0 – 200 |
| SkinThickness | Triceps skinfold thickness (mm) | 0 – 100 |
| Insulin | 2-Hour serum insulin (µU/mL) | 0 – 900 |
| BMI | Body mass index (kg/m²) | 0 – 70 |
| DiabetesPedigreeFunction | Diabetes pedigree function | 0 – 3 |
| Age | Age in years | 1 – 120 |

---

## 📁 Project Structure

```
diabetes_prediction/
│
├── backend/
│   ├── app.py                   ← Flask API (all endpoints)
│   ├── utils.py                 ← Helpers: validation, prediction logic
│   ├── train_model.py           ← Model training script (run once)
│   ├── model.pkl                ← Saved SVM model
│   ├── scaler.pkl               ← Saved StandardScaler
│   ├── requirements.txt         ← Python dependencies
│   ├── Procfile                 ← Render/Railway deployment config
│   ├── .gitignore
│   └── dataset/
│       ├── diabetes.csv         ← PIMA Indians dataset
│       └── sample_batch.csv     ← Sample CSV for testing
│
└── frontend.html                ← Standalone frontend UI
```

---

## 🚀 Quick Start (Run Locally)

### Prerequisites
- Python 3.8+
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/guptamanvi1106/-diabetes-prediction-system.git
cd -diabetes-prediction-system
```

### Step 2 — Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 3 — Train the model
```bash
python train_model.py
```
You will see:
```
Model Accuracy : 92.21%
[✓] Training complete.
```

### Step 4 — Start the API server
```bash
python app.py
```
API runs at: `http://localhost:5000`

### Step 5 — Open the frontend
Open `frontend.html` in your browser and set API URL to `http://localhost:5000`

---

## 🔌 API Endpoints

### GET `/health`
```json
{
  "status": "success",
  "service": "Diabetes Prediction API",
  "version": "1.0.0",
  "model_status": "loaded"
}
```

### POST `/predict`
**Request:**
```json
{
  "Pregnancies": 5,
  "Glucose": 166,
  "BloodPressure": 72,
  "SkinThickness": 19,
  "Insulin": 175,
  "BMI": 25.8,
  "DiabetesPedigreeFunction": 0.587,
  "Age": 51
}
```
**Response:**
```json
{
  "status": "success",
  "prediction": "Diabetic",
  "prediction_code": 1,
  "confidence": 0.87,
  "probabilities": {
    "not_diabetic": 0.13,
    "diabetic": 0.87
  }
}
```

### POST `/predict-batch`
Upload CSV file for bulk predictions. Add `?download=true` for CSV response.

### GET `/model-info`
Returns model metadata, accuracy, and scaler statistics.

### GET `/sample-input`
Returns a ready-to-use sample payload for testing.

---

## 📋 Sample CSV Format

```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
5,166,72,19,175,25.8,0.587,51
1,85,66,29,0,26.6,0.351,31
```

Sample file available at `backend/dataset/sample_batch.csv`

---

## ☁️ Deployment

### Backend → Render (Free)

| Field | Value |
|-------|-------|
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt && python train_model.py` |
| Start Command | `gunicorn app:app --bind 0.0.0.0:$PORT` |

### Frontend → Netlify (Free)
Drag and drop `frontend.html` onto [netlify.com](https://netlify.com) dashboard.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | API server port |
| `FLASK_DEBUG` | `false` | Enable debug mode |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**.
It is **not** a substitute for professional medical advice or diagnosis.

---

## 👤 Author

**Manvi Gupta**
- GitHub: [@guptamanvi1106](https://github.com/guptamanvi1106)
- Email: guptamanvi1106@gmail.com

---

## 📄 License

This project is licensed under the **MIT License**.

---

## ⭐ Support

If you found this helpful, give it a **star ⭐** on GitHub — it really helps!
