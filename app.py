from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load models
models = {
    "lstm": load_model("hinglish_lstm_model.keras"),
    "bilstm": load_model("hinglish_bilstm_model.keras"),
    "gru": load_model("hinglish_gru_model.keras"),
}

def predict_next_words(model, phrase):
    if not phrase:
        return ["No valid input"]

    # Convert text to sequences
    sequence = tokenizer.texts_to_sequences([phrase])

    if not sequence or not sequence[0]:  
        return ["No valid input"]

    sequence = np.array(sequence)

    # Predict next word probabilities
    predictions = model.predict(sequence)

    # Get top 3 predicted word indices
    predicted_indices = np.argsort(predictions[0])[-3:][::-1]  

    # Convert indices to words
    predicted_words = [tokenizer.index_word.get(idx, "unknown") for idx in predicted_indices]

    return predicted_words

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        phrase = data.get("phrase", "").strip()
        model_type = data.get("model", "lstm").lower()

        # Debug logs
        print(f"Model selected: {model_type}")
        print(f"Phrase received: {phrase}")

        if not phrase:
            return jsonify({"error": "Input text is empty"}), 400
        if model_type not in models:
            return jsonify({"error": "Invalid model type"}), 400

        model = models[model_type]
        predicted_words = predict_next_words(model, phrase)

        # Debug log
        print(f"Predictions: {predicted_words}")

        return jsonify({"input": phrase, "predictions": predicted_words})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)