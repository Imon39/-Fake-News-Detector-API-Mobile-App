import os
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(BASE_DIR, "Models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "Models", "tfidf_vectorizer.pkl")

model = None
vectorizer = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Success: Models loaded from Models folder!")
    else:
        print(f"❌ Error: Files not found at: {MODEL_PATH}")
        print("Current Directory Files:", os.listdir(BASE_DIR))
        if os.path.exists(os.path.join(BASE_DIR, "Models")):
            print("Models folder exists. Files inside:", os.listdir(os.path.join(BASE_DIR, "Models")))
except Exception as e:
    print(f"❌ Critical Error: {str(e)}")

@app.route("/")
def home():
    return "Fake News Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model files missing on server"}), 500
        
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' key"}), 400

        text = data["text"]
        vectorized = vectorizer.transform([text])
        prediction = model.predict(vectorized)[0]

       
        result = int(prediction)

        return jsonify({
            "is_fake": bool(result),
            "prediction_code": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
