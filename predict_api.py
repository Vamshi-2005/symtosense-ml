from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# ✅ Manual CORS (no flask-cors dependency needed)
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "SymtoSense ML API"})

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

print(f"✅ Model loaded. Symptoms supported: {columns}")

@app.route('/predict', methods=['GET', 'POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    # Support both GET (query params) and POST (JSON body)
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        symptoms_raw = data.get('symptoms', '')
    else:
        symptoms_raw = request.args.get('symptoms', '')

    if not symptoms_raw:
        return jsonify({"error": "No symptoms provided"}), 400

    # Parse + clean symptoms
    input_symptoms = [s.strip().lower().replace(' ', '_') for s in symptoms_raw.split(',')]

    # Build feature vector
    input_dict = {col: 0 for col in columns}
    matched = []
    unmatched = []
    for symptom in input_symptoms:
        if symptom in input_dict:
            input_dict[symptom] = 1
            matched.append(symptom)
        else:
            unmatched.append(symptom)

    if not matched:
        return jsonify({
            "error": "No recognized symptoms found",
            "supported_symptoms": columns,
            "provided": input_symptoms
        }), 400

    input_df = pd.DataFrame([input_dict])

    # Primary prediction
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    # Top 3 predictions
    top3_idx = np.argsort(probabilities)[::-1][:3]
    top3 = [
        {
            "disease": str(classes[i]),
            "confidence": round(float(probabilities[i]) * 100, 1)
        }
        for i in top3_idx if probabilities[i] > 0
    ]

    confidence = round(float(max(probabilities)) * 100, 1)

    return jsonify({
        "disease": str(prediction),
        "confidence": confidence,
        "top_predictions": top3,
        "matched_symptoms": matched,
        "unmatched_symptoms": unmatched
    })

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({"symptoms": columns})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)