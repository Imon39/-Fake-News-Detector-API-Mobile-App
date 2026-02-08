import os
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "Models", "tfidf_vectorizer.pkl")


model = None
vectorizer = None

def load_model():
    global model, vectorizer
    if model is None or vectorizer is None:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

@app.route("/")
def home():
    return "Fake News Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_model()
    except Exception as e:
        return jsonify({"error": f"Model loading failed: {str(e)}"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' key in request"}), 400

    text = data["text"]
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]

    return jsonify({
        "text_preview": text[:50] + "...",
        "is_fake": bool(prediction)
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
