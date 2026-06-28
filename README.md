# 🚂 Indian Railway Tracker

A Python/Flask web app to track Indian Railways in real-time.

🔗 **Live Demo:** https://your-railway-url.up.railway.app

---

## Features

- 🎫 **PNR Status** — Check ticket status with passenger-wise booking and seat details
- 🚉 **Train Search** — Search trains between any two stations with timings and running days
- 📍 **Live Train Status** — Real-time train location, delay info and full station-wise timeline

---

## APIs Used

- **PNR Status** → IRCTC Indian Railway PNR Status (RapidAPI)
- **Train Search & Live Status** → Rail Radar API (railradar.in)

---

## Tech Stack

Python · Flask · HTML · CSS · JavaScript · REST APIs

---

## Local Setup

1. Clone: `git clone https://github.com/Saransh-Saurav/railway-pnr-checker.git`
2. Install: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add API keys
4. Run: `python app.py`
5. Open: `http://localhost:5000`

---

## Environment Variables
RAPIDAPI_KEY=your_rapidapi_key

RAILAPI_KEY=your_railradar_key


## Disclaimer

This project uses an unofficial third-party API for educational purposes only.
Not affiliated with Indian Railways or IRCTC.
For official PNR status, visit [indianrail.gov.in](https://www.indianrail.gov.in).

Replace your-railway-url with your actual Railway URL.
git add .
git commit -m "Update README"
git push
