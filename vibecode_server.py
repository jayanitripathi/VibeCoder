from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv("environment.env")

HF_API_KEY = os.getenv("HF_API_KEY")

app = Flask(__name__)

@app.route("/vibecode-webhook", methods=["POST"])
def vibecode_webhook():
    print("\n--- Incoming Request ---")
    print("Raw data:", request.data)

    data = request.get_json(silent=True)
    print("Parsed JSON:", data)

    transcript = None

    if isinstance(data, dict) and "segments" in data:
        segments = data["segments"]
        if isinstance(segments, list) and len(segments) > 0:
            transcript = segments[0].get("text")

    if not transcript:
        return jsonify({"error": "No transcript found"}), 400

    print("\n[Transcript Received]")
    print(transcript)

    # Send to Hugging Face emotion analysis model
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": transcript}
        )

        if response.status_code != 200:
            print("\n[HF API ERROR]", response.status_code, response.text)
            return jsonify({
                "error": "HF API error",
                "status_code": response.status_code,
                "message": response.text
            }), 500

        emotions = response.json()
        print("\n[Analysis Result]")
        print(emotions)

        return jsonify({"message": "Emotion analyzed", "emotions": emotions})

    except Exception as e:
        print("\n[Exception during HuggingFace call]", str(e))
        return jsonify({"error": "Unexpected error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5050)
