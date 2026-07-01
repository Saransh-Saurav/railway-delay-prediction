import os
import json
import joblib
import numpy as np
import requests
from datetime import date, datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
RAILAPI_KEY  = os.getenv("RAILAPI_KEY", "")

PNR_HOST  = "irctc-indian-railway-pnr-status.p.rapidapi.com"
RAIL_BASE = "https://api.railradar.in/v1"

# -- Load trained delay model (optional – app works without it) ---------------
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
_model     = None
_meta      = None

_model_error = None

def _load_model():
    global _model, _meta, _model_error
    model_path = os.path.join(_MODEL_DIR, "delay_model.joblib")
    meta_path  = os.path.join(_MODEL_DIR,  "meta.json")
    if os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            _model = joblib.load(model_path)
            with open(meta_path, encoding="utf-8") as f:
                _meta = json.load(f)
            print("[INFO] Delay prediction model loaded OK.")
        except Exception as e:
            _model_error = f"Load Exception: {str(e)}"
            print(f"[WARN] Could not load delay model: {e}")
    else:
        _model_error = f"Files missing. model_path exists: {os.path.exists(model_path)}, meta_path exists: {os.path.exists(meta_path)}. Looking in {_MODEL_DIR}"

_load_model()
# -----------------------------------------------------------------------------


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
    to_code   = request.args.get("to",   "").strip()
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


@app.route("/model_status")
def model_status():
    """Tell the frontend whether the ML model is ready."""
    return jsonify({"ready": _model is not None, "error": _model_error})


def _delay_status(delay_minutes):
    """Convert delay minutes into the UI status bucket."""
    if delay_minutes <= 5:
        return "On Time", "green"
    if delay_minutes <= 30:
        return "Slightly Delayed", "orange"
    if delay_minutes <= 90:
        return "Moderately Delayed", "amber"
    return "Highly Delayed", "red"


def _live_delay_for_station(train_no, station_code):
    """Return live delay for today's train when RailRadar has it."""
    if not RAILAPI_KEY:
        return None

    live_payload = rail_get(f"/trains/{train_no}/live")
    data = live_payload.get("data", live_payload)
    route = data.get("route", [])
    station = next(
        (s for s in route if str(s.get("stationCode", "")).upper() == station_code),
        None,
    )

    if station:
        for key in ("delayArrival", "delayDeparture", "delayMinutes"):
            if station.get(key) is not None:
                return {
                    "delay": float(station[key]),
                    "source": "live_station_actual",
                    "station_name": station.get("stationName"),
                }

    if data.get("delayMinutes") is not None:
        return {
            "delay": float(data["delayMinutes"]),
            "source": "live_train_current",
            "station_name": station.get("stationName") if station else None,
        }

    return None


@app.route("/predict_delay")
def predict_delay():
    """Return live delay for today or a forecast for future dates."""
    if _model is None or _meta is None:
        return jsonify({"error": "Model not trained yet. Please run train_delay_model.py first."}), 503

    train_no     = request.args.get("train", "").strip()
    station_code = request.args.get("station", "").strip().upper()
    journey_date = request.args.get("date", date.today().isoformat()).strip()

    if not train_no or not station_code:
        return jsonify({"error": "train and station are required."}), 400

    try:
        sched_lookup = _meta.get("sched_lookup", {})
        type_map     = _meta.get("type_map", {})
        zone_map     = _meta.get("zone_map", {})
        train_map    = _meta.get("train_map", {})
        train_avg    = _meta.get("train_avg", {})

        # Parse date features
        dt          = datetime.strptime(journey_date, "%Y-%m-%d")
        requested_day = dt.date()
        today = date.today()
        day_of_week = dt.weekday()    # 0=Mon
        month       = dt.month

        if requested_day < today:
            return jsonify({
                "error": (
                    f"Past dates are not predicted. {journey_date} is before {today.isoformat()}, "
                    "and this app does not yet have a historical actual-delay lookup for past journeys."
                )
            }), 400

        # Get schedule info for this train+station
        train_sched = sched_lookup.get(str(train_no), {})
        # Try exact station code first, then iterate for partial match
        sched_info = train_sched.get(station_code, None)
        if sched_info is None:
            # try scanning values for a match against station code
            for k, v in train_sched.items():
                if k.startswith(station_code):
                    sched_info = v
                    break
        if sched_info is None:
            return jsonify({"error": f"Station {station_code} is not on the route for Train {train_no}."}), 404

        station_no = int(sched_info.get("station_no", 5))
        distance   = float(sched_info.get("distance",   0))
        sched_hour = float(sched_info.get("sched_hour", 12))

        if requested_day == today:
            live_delay = _live_delay_for_station(train_no, station_code)
            if live_delay is not None:
                actual_delay = round(live_delay["delay"], 1)
                if distance == 0 or station_no == 1:
                    actual_delay = 0.0
                status, color = _delay_status(actual_delay)
                return jsonify({
                    "train_no": train_no,
                    "station": station_code,
                    "date": journey_date,
                    "predicted_delay": actual_delay,
                    "status": status,
                    "color": color,
                    "day_of_week": dt.strftime("%A"),
                    "mode": "live_actual",
                    "result_label": "Live Delay",
                    "value_label": "Actual Delay",
                    "source_note": "Using today's live running status from RailRadar.",
                })

        # Encode train type (look up from train_map → default 0)
        train_enc = int(train_map.get(str(train_no), 0))
        # Encode station zone via zone_map (we don't have zone for code, use 0)
        zone_enc  = int(zone_map.get("UNKNOWN", 0))
        # Encode type_code
        type_enc  = int(type_map.get("PASS-TRAINS", 0))

        features = np.array([[train_enc, station_no, distance,
                               sched_hour, day_of_week, month,
                               type_enc, zone_enc]], dtype=float)

        predicted_delay = float(_model.predict(features)[0])
        predicted_delay = round(predicted_delay, 1)

        # Provide fallback blending: if train is in history average
        if str(train_no) in train_avg:
            hist_avg = train_avg[str(train_no)]
            # Blend: 70% model, 30% historical mean
            predicted_delay = round(0.7 * predicted_delay + 0.3 * hist_avg, 1)

        # Origin station usually departs on time
        if distance == 0 or station_no == 1:
            predicted_delay = 0.0

        status, color = _delay_status(predicted_delay)

        return jsonify({
            "train_no":        train_no,
            "station":         station_code,
            "date":            journey_date,
            "predicted_delay": predicted_delay,
            "status":          status,
            "color":           color,
            "day_of_week":     dt.strftime("%A"),
            "mode":            "forecast",
            "result_label":    "Forecast Result",
            "value_label":     "Expected Delay",
            "source_note":     "Forecast based on historical patterns for the selected future journey date.",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
