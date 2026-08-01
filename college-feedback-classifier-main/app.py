"""
College Feedback Classifier — Flask Web Application
Trains a TF-IDF + Multinomial Naive Bayes classifier on student_feedback.csv
and serves a premium web UI for interactive feedback classification.
"""

import os
import re
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from collections import Counter

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────
app = Flask(__name__)

# ──────────────────────────────────────────────
# Load Dataset & Train Model
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "student_feedback.csv")

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["feedback_text", "category"])
df["category"] = df["category"].str.strip()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["feedback_text"], df["category"],
    test_size=0.3, random_state=42, stratify=df["category"]
)

# Build pipeline: TF-IDF → Naive Bayes
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )),
    ("clf", MultinomialNB(alpha=0.1))
])
pipeline.fit(X_train, y_train)

# Evaluate on test set (logged at startup)
y_pred_test = pipeline.predict(X_test)
print("\n══════════════════════════════════════════")
print("  Model Training Complete!")
print("══════════════════════════════════════════")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples:     {len(X_test)}")
print(f"  Accuracy:         {(y_pred_test == y_test).mean():.1%}")
print("══════════════════════════════════════════\n")
print(classification_report(y_test, y_pred_test))


# ──────────────────────────────────────────────
# Sentiment Analysis (keyword-based)
# ──────────────────────────────────────────────
POSITIVE_WORDS = {
    "good", "great", "excellent", "well", "best", "amazing", "modern",
    "comfortable", "spacious", "clean", "helpful", "friendly", "supportive",
    "innovative", "interactive", "enthusiastic", "inspiring", "motivated",
    "encouraging", "constructive", "vibrant", "reassuring", "fair",
    "well-maintained", "well-organized", "well-structured", "well-furnished",
    "well-qualified", "well-trained", "well-equipped", "knowledgeable",
    "approachable", "dedicated", "experienced", "patient", "cooperative",
    "punctual", "engaging", "manageable", "relevant", "variety"
}

NEGATIVE_WORDS = {
    "bad", "poor", "slow", "difficult", "complicated", "confusing",
    "unclear", "insufficient", "inadequate", "outdated", "overcrowded",
    "tedious", "lengthy", "not", "needs", "improvement", "misplace",
    "break", "crashes", "lacks", "crowded", "strict", "boring",
    "ineffective", "not enough", "worse", "terrible", "horrible"
}


def analyze_sentiment(text: str) -> dict:
    """Simple keyword-based sentiment analysis."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)

    if pos_count > neg_count:
        sentiment = "positive"
        score = min(0.95, 0.6 + (pos_count - neg_count) * 0.1)
    elif neg_count > pos_count:
        sentiment = "negative"
        score = min(0.95, 0.6 + (neg_count - pos_count) * 0.1)
    else:
        sentiment = "neutral"
        score = 0.5

    return {"sentiment": sentiment, "confidence": round(score, 2)}


# ──────────────────────────────────────────────
# Routes — Pages
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ──────────────────────────────────────────────
# Routes — API
# ──────────────────────────────────────────────
@app.route("/api/classify", methods=["POST"])
def classify():
    """Classify a single feedback text."""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Category prediction with probabilities
    category = pipeline.predict([text])[0]
    probas = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    category_scores = {
        cls: round(float(prob), 4)
        for cls, prob in zip(classes, probas)
    }

    # Sentiment
    sentiment_result = analyze_sentiment(text)

    return jsonify({
        "text": text,
        "category": category,
        "category_confidence": category_scores[category],
        "category_scores": category_scores,
        "sentiment": sentiment_result["sentiment"],
        "sentiment_confidence": sentiment_result["confidence"]
    })


@app.route("/api/batch-classify", methods=["POST"])
def batch_classify():
    """Classify multiple feedback texts."""
    data = request.get_json()
    texts = data.get("texts", [])

    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    results = []
    for text in texts:
        text = text.strip()
        if not text:
            continue

        category = pipeline.predict([text])[0]
        probas = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
        category_scores = {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probas)
        }
        sentiment_result = analyze_sentiment(text)

        results.append({
            "text": text,
            "category": category,
            "category_confidence": category_scores[category],
            "category_scores": category_scores,
            "sentiment": sentiment_result["sentiment"],
            "sentiment_confidence": sentiment_result["confidence"]
        })

    # Summary stats
    categories = [r["category"] for r in results]
    sentiments = [r["sentiment"] for r in results]
    cat_counts = dict(Counter(categories))
    sent_counts = dict(Counter(sentiments))

    return jsonify({
        "results": results,
        "summary": {
            "total": len(results),
            "category_distribution": cat_counts,
            "sentiment_distribution": sent_counts
        }
    })


@app.route("/api/stats")
def stats():
    """Get dataset statistics."""
    cat_counts = df["category"].value_counts().to_dict()
    total = len(df)

    # Compute sentiment distribution for the whole dataset
    sentiments = [analyze_sentiment(t)["sentiment"] for t in df["feedback_text"]]
    sent_counts = dict(Counter(sentiments))

    # Model performance
    accuracy = float((y_pred_test == y_test).mean())
    report = classification_report(y_test, y_pred_test, output_dict=True)

    return jsonify({
        "total_samples": total,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "category_distribution": cat_counts,
        "sentiment_distribution": sent_counts,
        "model_accuracy": round(accuracy, 4),
        "classification_report": report
    })


@app.route("/api/dataset")
def dataset():
    """Get paginated dataset entries."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    category_filter = request.args.get("category", "all")

    filtered_df = df.copy()
    if category_filter != "all":
        filtered_df = filtered_df[filtered_df["category"] == category_filter]

    total = len(filtered_df)
    start = (page - 1) * per_page
    end = start + per_page
    page_data = filtered_df.iloc[start:end]

    entries = []
    for _, row in page_data.iterrows():
        sentiment = analyze_sentiment(row["feedback_text"])
        entries.append({
            "text": row["feedback_text"],
            "category": row["category"],
            "sentiment": sentiment["sentiment"]
        })

    return jsonify({
        "entries": entries,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page
    })


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Starting College Feedback Classifier...")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
