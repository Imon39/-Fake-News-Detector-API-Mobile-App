import os
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_news_model.pkl") 
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Model and Vectorizer loaded successfully!")
    else:
        print("❌ Error: Model files not found at defined paths.")
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")

@app.route("/")
def home():
    return "Fake News Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' key in request"}), 400

        text = data["text"]
        
        vectorized = vectorizer.transform([text])
        prediction = model.predict(vectorized)[0]

        result = int(prediction) 

        return jsonify({
            "text_preview": text[:50] + "...",
            "is_fake": bool(result),
            "prediction_code": result
        })

    except NameError:
        return jsonify({"error": "Model not loaded on server. Check file paths."}), 500
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
