from flask import Flask, request, jsonify
import joblib
import numpy as np

# load trained model
model = joblib.load("disease_model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    fever = data.get("fever", 0)
    cough = data.get("cough", 0)
    headache = data.get("headache", 0)
    fatigue = data.get("fatigue", 0)
    nausea = data.get("nausea", 0)
    vomiting = data.get("vomiting", 0)

    features = np.array([[fever, cough, headache, fatigue, nausea, vomiting]])

    prediction = model.predict(features)

    return jsonify({
        "prediction": prediction[0]
    })


if __name__ == "__main__":
    app.run(port=5000)