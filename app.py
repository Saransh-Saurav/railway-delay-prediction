import os
import requests
from datetime import date
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
RAILAPI_KEY  = os.getenv("RAILAPI_KEY", "")

PNR_HOST = "irctc-indian-railway-pnr-status.p.rapidapi.com"
RAIL_BASE = "https://api.railradar.in/v1"


def rail_get(endpoint):
    response = requests.get(
        f"{RAIL_BASE}{endpoint}",
        headers={"Authorization": f"Bearer {RAILAPI_KEY}"},
        timeout=10
    )
    response.raise_for_status()
    return response.json()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/check")
def check_pnr():
    pnr = request.args.get("pnr", "").strip()
    if not pnr or not pnr.isdigit() or len(pnr) != 10:
        return jsonify({"error": "Invalid PNR number."}), 400
    if not RAPIDAPI_KEY:
        return jsonify({"error": "RAPIDAPI_KEY not configured."}), 500
    try:
        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,
            "x-rapidapi-host": PNR_HOST,
            "Content-Type": "application/json",
        }
        response = requests.get(
            f"https://{PNR_HOST}/getPNRStatus/{pnr}",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/trains_between")
def trains_between():
    from_code = request.args.get("from", "").strip()
    to_code   = request.args.get("to", "").strip()
    if not from_code or not to_code:
        return jsonify({"error": "from and to are required."}), 400
    if not RAILAPI_KEY:
        return jsonify({"error": "RAILAPI_KEY not configured."}), 500
    try:
        today = date.today().isoformat()
        return jsonify(rail_get(f"/trains/between/{from_code}/{to_code}?date={today}"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/live_status")
def live_status():
    train_no = request.args.get("train", "").strip()
    if not train_no:
        return jsonify({"error": "train number is required."}), 400
    if not RAILAPI_KEY:
        return jsonify({"error": "RAILAPI_KEY not configured."}), 500
    try:
        return jsonify(rail_get(f"/trains/{train_no}/live"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
