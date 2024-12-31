from flask import Flask, jsonify, request, render_template
import joblib
import tweepy
from model.emotion_classifier import EmotionClassifier, TextPreprocessor

app = Flask(__name__, 
    static_folder='static',
    template_folder='templates'
)

emotion_map = {
    "0" : "sadness",
    "1" : 'joy',
    "2" : 'love',
    "3" : 'anger',
    "4" : 'fear'
}
# Constants
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAFogxwEAAAAATzyEHo2r5OJCoq9uy0i7Vea9BLc%3DRLFOXEN5CkKqUqsu03xGGx8WMFPSwmMBInuHJ7GcPzZYxDe0DK"

# Initialize Tweepy client
try:
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
except Exception as e:
    print(f"Failed to initialize Tweepy client: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test_model", methods=["POST"])
def test_model():
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({"error": "No text provided"}), 400
        classifier = EmotionClassifier()
        classifier.load_model('model/emotion_classifier.joblib')
        result = classifier.predict_emotion(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to predict sentiments of recent tweets."""
    try:
        # Parse incoming JSON request
        data = request.get_json()
        query = data.get("query")
        max_results = data.get("max_results", 10)
        min_likes = data.get("min_likes", 0)

        # Validate query parameter
        if not query:
            return jsonify({"error": "The 'query' parameter is required."}), 400

        # Fetch recent tweets
        tweets = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=['created_at', 'text', 'author_id', 'public_metrics']
        )
        classifier = EmotionClassifier()
        classifier.load_model('model/emotion_classifier.joblib')
        # Process and structure tweet data
        tweet_data = [
            {
                "id": tweet.id,
                "text": tweet.text,
                "created_at": tweet.created_at,
                "author_id": tweet.author_id,
                "sentiment": classifier.predict_emotion(tweet.text),
                "url": f"https://twitter.com/{tweet.author_id}/status/{tweet.id}",
            }
            for tweet in (tweets.data or [])
        ]

        return jsonify(tweet_data)

    except tweepy.TooManyRequests as e:
        rate_limit_details = {
            "error": "Rate limit exceeded. Please try again later.",
            "reset_time": e.response.headers.get("x-rate-limit-reset"),
            "limit": e.response.headers.get("x-rate-limit-limit"),
            "remaining": e.response.headers.get("x-rate-limit-remaining")
        }
        return jsonify(rate_limit_details), 429
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)