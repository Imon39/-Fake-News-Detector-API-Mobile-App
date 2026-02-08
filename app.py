import os
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

MODEL_PATH = os.path.join("Models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join("Models", "tfidf_vectorizer.pkl")

# Load model and vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    print("Error: model not found!")

@app.route('/')
def home():
    return "Fake News Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' key in request"}), 400
    
    text = data.get("text")
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    

    return jsonify({
        "text_preview": text[:50] + "...",
        "is_fake": bool(prediction)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
