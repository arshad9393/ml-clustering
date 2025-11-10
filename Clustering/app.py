from flask import Flask, request, jsonify, render_template
import os, joblib, numpy as np, pandas as pd

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))
gmm = joblib.load(os.path.join(MODELS_DIR, "gmm.pkl"))

FEATURE_ORDER = ["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"]
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or request.form.to_dict()
    if isinstance(data, dict) and not all(k in data for k in FEATURE_ORDER):
        try:
            df = pd.DataFrame([data])
        except Exception:
            return jsonify({"error":"invalid input"}), 400
    else:
        df = pd.DataFrame([data])
    try:
        X = df[FEATURE_ORDER].astype(float).fillna(0)
    except Exception:
        return jsonify({"error":"missing or invalid features"}), 400
    X_log = np.log1p(X)
    X_scaled = scaler.transform(X_log)
    k_pred = kmeans.predict(X_scaled).tolist()
    g_pred = gmm.predict(X_scaled).tolist()
    return jsonify({"kmeans": k_pred, "gmm": g_pred})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
