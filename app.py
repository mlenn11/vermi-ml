from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load worm health model
with open("worm_model.pkl", "rb") as f:
    saved = pickle.load(f)
    model    = saved['model']
    features = saved['features']

# Trend memory — store last 3 readings
history = []

def stress_score(temp, hum, soil, ph):
    score = 0
    if not (20 <= temp <= 30): score += 1
    if not (60 <= hum <= 80):  score += 1
    if not (60 <= soil <= 80): score += 1
    if not (6.0 <= ph <= 7.5): score += 1
    return score

def get_prediction_message(label, confidence, temp, hum, soil, ph):
    messages = {
        "Thriving": {
            "emoji": "🟢",
            "summary": "Worms are thriving! All conditions are perfect.",
            "detail": "Your compost environment is ideal. Worms are active, healthy and reproducing well. Keep maintaining these conditions."
        },
        "Healthy": {
            "emoji": "🟡",
            "summary": "Worms are healthy with minor concerns.",
            "detail": "Worms are doing well but some conditions are slightly off the ideal range. Monitor closely and consider small adjustments."
        },
        "Stressed": {
            "emoji": "🟠",
            "summary": "Worms are stressed — conditions need attention.",
            "detail": "Multiple parameters are outside ideal range. Worms may slow down activity and reproduction. Take corrective action soon."
        },
        "At Risk": {
            "emoji": "🔴",
            "summary": "Worms are at risk — immediate action recommended.",
            "detail": "Conditions are significantly deteriorating. Worm health and survival are at risk. Act now to prevent worm die-off."
        },
        "Critical": {
            "emoji": "🚨",
            "summary": "Worms in critical danger — act immediately!",
            "detail": "Conditions are dangerous for worm survival. Without immediate intervention, worm population may collapse."
        }
    }
    return messages.get(label, messages["Stressed"])

def get_issues_and_actions(temp, hum, soil, ph, label):
    issues = []
    actions = []
    auto_actions = []

    if label in ["Critical", "At Risk"]:
        if temp >= 35:
            issues.append("Temperature dangerously high (≥35°C)")
            auto_actions.append("🔓 Lid opens for 10 seconds then closes automatically")
        if hum < 50:
            issues.append("Humidity critically low (<50%)")
            auto_actions.append("💧 Water pump activates for 6 seconds")
            auto_actions.append("⏳ 3 second delay")
            auto_actions.append("⚙️ Propeller mixer activates for 6 seconds")
        if hum > 85:
            issues.append("Humidity critically high (>85%)")
            auto_actions.append("🔓 Lid opens → 2s delay → Mixer 6s → 2s delay → Lid closes")
        if soil < 40:
            issues.append("Soil moisture critically low (<40%)")
            if not any("pump" in a.lower() for a in auto_actions):
                auto_actions.append("💧 Water pump activates for 6 seconds")
                auto_actions.append("⚙️ Propeller mixer activates for 6 seconds")
        if soil > 85:
            issues.append("Soil moisture critically high (>85%)")
            if not any("lid" in a.lower() for a in auto_actions):
                auto_actions.append("🔓 Lid opens → 2s delay → Mixer 6s → 2s delay → Lid closes")
        if ph < 5.5:
            issues.append("pH dangerously acidic (<5.5)")
            # No auto action — suggestion only
            actions.append("🪱 Add crushed eggshells or agricultural lime immediately to neutralize acidity")
        if ph > 8.0:
            issues.append("pH dangerously alkaline (>8.0)")
            # No auto action — suggestion only
            actions.append("🪱 Add coffee grounds or citrus peels immediately to reduce alkalinity")

    elif label in ["Stressed", "Healthy"]:
        if temp < 20:
            issues.append("Temperature too low (<20°C)")
            actions.append("🔒 Keep lid closed to retain heat")
        elif temp > 30:
            issues.append("Temperature getting high (30–35°C)")
            actions.append("🔓 Open lid to reduce temperature")
        if 50 <= hum <= 59:
            issues.append("Humidity slightly low (50–59%)")
            actions.append("💧 Activate water pump to increase moisture")
        elif 81 <= hum <= 85:
            issues.append("Humidity slightly high (81–85%)")
            actions.append("⚙️ Activate propeller to improve oxygen circulation")
        if 40 <= soil <= 59:
            issues.append("Soil moisture slightly low (40–59%)")
            if not any("pump" in a.lower() for a in actions):
                actions.append("💧 Activate water pump to increase moisture")
        elif 81 <= soil <= 85:
            issues.append("Soil moisture slightly high (81–85%)")
        if 5.5 <= ph <= 5.9:
            issues.append("pH slightly acidic (5.5–5.9)")
            actions.append("🪱 You might want to add crushed eggshells or lime to keep your compost natural")
        elif 7.6 <= ph <= 8.0:
            issues.append("pH slightly alkaline (7.6–8.0)")
            actions.append("🪱 You might want to add coffee grounds or citrus peels to keep your compost natural")

    return issues, actions, auto_actions


@app.route("/predict", methods=["POST"])
def predict():
    global history
    try:
        body = request.get_json()
        temp = float(body["temperature"])
        hum  = float(body["humidity"])
        soil = float(body["moisture"])
        ph   = float(body["ph"])

        # Add to history
        history.append({'temp': temp, 'hum': hum, 'soil': soil, 'ph': ph})
        if len(history) > 3:
            history = history[-3:]

        # Compute trends
        if len(history) >= 2:
            temp_trend = history[-1]['temp'] - history[-2]['temp']
            hum_trend  = history[-1]['hum']  - history[-2]['hum']
            soil_trend = history[-1]['soil'] - history[-2]['soil']
            ph_trend   = history[-1]['ph']   - history[-2]['ph']
        else:
            temp_trend = hum_trend = soil_trend = ph_trend = 0

        # Rolling averages
        temp_roll = sum(h['temp'] for h in history) / len(history)
        hum_roll  = sum(h['hum']  for h in history) / len(history)
        soil_roll = sum(h['soil'] for h in history) / len(history)

        ss = stress_score(temp, hum, soil, ph)

        row = pd.DataFrame([{
            'temperature':  temp,
            'humidity':     hum,
            'moisture':     soil,
            'ph':           ph,
            'temp_trend':   temp_trend,
            'hum_trend':    hum_trend,
            'soil_trend':   soil_trend,
            'ph_trend':     ph_trend,
            'temp_roll':    temp_roll,
            'hum_roll':     hum_roll,
            'soil_roll':    soil_roll,
            'stress_score': ss
        }])

        label      = model.predict(row)[0]
        proba      = model.predict_proba(row)[0]
        confidence = round(float(max(proba)) * 100, 1)
        msg        = get_prediction_message(label, confidence, temp, hum, soil, ph)
        issues, actions, auto_actions = get_issues_and_actions(temp, hum, soil, ph, label)

        # Trend interpretation
        trend_msg = ""
        if len(history) >= 2:
            if temp_trend > 0.5:
                trend_msg = "⚠️ Temperature rising"
            elif temp_trend < -0.5:
                trend_msg = "✅ Temperature cooling down"
            if soil_trend < -5:
                trend_msg += " | ⚠️ Moisture dropping fast"
            elif soil_trend > 5:
                trend_msg += " | ✅ Moisture improving"

        return jsonify({
            "worm_health":   label,
            "confidence":    confidence,
            "emoji":         msg["emoji"],
            "summary":       msg["summary"],
            "detail":        msg["detail"],
            "trend":         trend_msg,
            "issues":        issues,
            "actions":       actions,
            "auto_actions":  auto_actions,
            "trends": {
                "temperature": round(temp_trend, 2),
                "humidity":    round(hum_trend, 2),
                "moisture":    round(soil_trend, 2),
                "ph":          round(ph_trend, 2)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "GradientBoosting Worm Health Predictor",
        "accuracy": "98.55%",
        "trained_on": "342 real vermicompost readings"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)