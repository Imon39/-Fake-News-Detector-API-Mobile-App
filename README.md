
📰 Fake News Detection API & Mobile App

This project leverages **Machine Learning (NLP)** to detect whether a news article is fake or real. It includes a trained model, a **Flask API** for deployment, and an **Android application** for end-users.

---

## 🚀 Features

* **ML Model:** Built using **Scikit-Learn** with a high accuracy score.
* **Flask API:** A RESTful API to provide real-time predictions.
* **Mobile App:** An `.apk` file ready to be installed on Android devices.
* **Cloud Ready:** Includes `Procfile` for easy deployment on platforms like **Render** or **Heroku**.

## 📂 Project Structure

| File/Folder | Description |
| --- | --- |
| `app.py` | The main Flask application script. |
| `fake_news_model.pkl` | The saved Machine Learning model. |
| `tfidf_vectorizer.pkl` | The TF-IDF vectorizer for text processing. |
| `Fake News Detector.apk` | The Android application file. |
| `fake_news_detection_code.ipynb` | Jupyter Notebook containing model training & analysis. |
| `requirements.txt` | List of Python dependencies. |
| `Procfile` | Configuration for cloud hosting (Render/Heroku). |

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Imon39/fake-news-api.git
cd fake-news-api

```

### 2. Install Dependencies

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt

```

### 3. Run the API locally

```bash
python app.py

```

The server will start at `http://127.0.0.1:5000`.

---

## 📡 API Usage

### **Endpoint:** `/predict`

**Method:** `POST`

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "text": "Your news headline or article content here..."
}

```

**Response:**

```json
{
  "is_fake": true,
  "text_preview": "Your news headline..."
}

```

---

## 🤖 Model Details

The model was trained using a dataset of labeled real and fake news.

* **Preprocessing:** Tokenization, Stop-word removal, and TF-IDF Vectorization.
* **Algorithm:** Logistic Regression / Passive Aggressive Classifier.

## 📱 Mobile App

You can find the pre-built Android app in the root directory: `Fake News Detector.apk`. Simply download it to your Android device and install it to test the detection on the go.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/Imon39/fake-news-api/issues).
