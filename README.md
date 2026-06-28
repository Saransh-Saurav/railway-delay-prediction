# 🚂 Indian Railway Tracker

A full-featured Indian Railway tracking web app with **PNR status checking**, **train search**, **live running status**, and **ML-powered delay prediction**. Built with Flask and deployed on Vercel.

🔗 **Live Demo**: [railway-delay-prediction.vercel.app](https://railway-delay-prediction.vercel.app)

---

## ✨ Features

### 🎫 PNR Status Check
- Enter any 10-digit PNR number to get real-time booking status
- Displays train info, journey details, boarding/destination stations
- Shows passenger-wise booking and current status (CNF / WL / RAC)

### 🔍 Train Search
- Search trains between any two stations across India
- Autocomplete station names with station codes
- View train numbers, timings, and running days

### 📍 Live Running Status
- Track any train's live running status
- Search by train number or train name
- See current position and delay information

### ⚡ Delay Prediction (ML-Powered)
- Predict expected delay for any train at any station on a given date
- Uses a trained **scikit-learn** model with features like:
  - Train number & type
  - Station sequence & distance from origin
  - Scheduled arrival hour
  - Day of week & month
- Color-coded results: 🟢 On Time · 🟠 Slightly Delayed · 🟡 Moderately Delayed · 🔴 Highly Delayed

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python, Flask |
| **Frontend** | HTML, CSS, JavaScript |
| **ML Model** | scikit-learn, joblib, NumPy |
| **APIs** | RapidAPI (PNR Status), RailRadar API |
| **Deployment** | Vercel (Serverless Python) |

---

## 📁 Project Structure

```
railway-delay-prediction/
├── app.py                     # Flask application (routes & API logic)
├── train_delay_model.py       # Script to train the delay prediction model
├── download_delay_dataset.py  # Script to download training data
├── requirements.txt           # Python dependencies
├── vercel.json                # Vercel deployment configuration
├── Procfile                   # Process file for Heroku (optional)
├── runtime.txt                # Python runtime version
├── .env.example               # Environment variable template
├── model/
│   ├── delay_model.joblib     # Trained ML model
│   ├── meta.json              # Model metadata & station schedule lookup
│   ├── le_train.joblib        # Label encoder - train numbers
│   ├── le_type.joblib         # Label encoder - train types
│   └── le_zone.joblib         # Label encoder - railway zones
├── templates/
│   └── index.html             # Frontend UI
└── static/                    # Static assets
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- API keys from [RapidAPI](https://rapidapi.com/) and [RailRadar](https://railradar.in/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Saransh-Saurav/railway-delay-prediction.git
   cd railway-delay-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your API keys:
   ```
   RAPIDAPI_KEY=your_rapidapi_key_here
   RAILAPI_KEY=your_railapi_key_here
   ```

5. **Run the app**
   ```bash
   python app.py
   ```
   Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🤖 Training the Delay Model

The delay prediction model can be retrained with fresh data:

```bash
# Step 1: Download the latest delay dataset
python download_delay_dataset.py

# Step 2: Train the model
python train_delay_model.py
```

The trained model and metadata are saved to the `model/` directory.

---

## 🌐 Deployment on Vercel

1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com) → **Add New Project** → Import your repo
3. Add environment variables in the Vercel dashboard:
   - `RAPIDAPI_KEY`
   - `RAILAPI_KEY`
4. Click **Deploy**

The `vercel.json` file handles all routing configuration automatically.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/check?pnr=XXXXXXXXXX` | GET | Check PNR status |
| `/trains_between?from=NDLS&to=HWH` | GET | Search trains between stations |
| `/live_status?train=12301` | GET | Get live running status |
| `/predict_delay?train=12301&station=CNB&date=2025-01-15` | GET | Predict delay (ML) |
| `/model_status` | GET | Check if ML model is loaded |
| `/health` | GET | Health check |

---

## 📸 Screenshots

| PNR Status | Train Search | Delay Prediction |
|:---:|:---:|:---:|
| Check real-time PNR status | Find trains between stations | ML-powered delay forecasting |

---

## ⚠️ Disclaimer

This is an **unofficial** project built for educational purposes. It is **not affiliated** with Indian Railways, IRCTC, or any government body. For official information, visit [indianrail.gov.in](https://www.indianrail.gov.in).

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Saransh Saurav**
- GitHub: [@Saransh-Saurav](https://github.com/Saransh-Saurav)

---

⭐ If you found this useful, give it a star!
