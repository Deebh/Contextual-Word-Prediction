from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load tokenizer for traditional models
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load traditional models
models = {
    "lstm": load_model("hinglish_lstm_model.keras"),
    "bilstm": load_model("hinglish_bilstm_model.keras"),
    "gru": load_model("hinglish_gru_model.keras"),
    "bilstm_gru": load_model("hinglish_bilstm_gru_model.keras"),
}

# Load GPT-2 model and tokenizer from local directory
gpt2_model_path = "gpt2-hinglish-finetuned" 
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)

# Configure tokenizer settings
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_tokenizer.padding_side = "left"  # For better generation with left-padding

def predict_next_words(model, phrase):
    if not phrase:
        raise ValueError("Input text is empty")

    sequence = tokenizer.texts_to_sequences([phrase])
    if not sequence or not sequence[0]:  
        raise ValueError("Input contains no valid tokens")

    sequence = np.array(sequence)
    predictions = model.predict(sequence)
    predicted_indices = np.argsort(predictions[0])[-3:][::-1]  
    predicted_words = [tokenizer.index_word.get(idx, "unknown") for idx in predicted_indices]
    return predicted_words

def predict_gpt2(phrase, top_n=3):
    try:
        # Encode the input text with attention mask
        inputs = gpt2_tokenizer(phrase, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Generate predictions with more appropriate settings for Hinglish
        output = gpt2_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,  # Generate just the next few tokens
            num_return_sequences=top_n,
            do_sample=True,
            top_k=30,  # Lower top_k for more focused predictions
            top_p=0.9,  # Slightly lower top_p
            temperature=0.7,  # Lower temperature for less random predictions
            pad_token_id=gpt2_tokenizer.eos_token_id,
            no_repeat_ngram_size=2,  # Avoid repeating 2-grams
            early_stopping=True
        )
        
        # Decode only the new tokens (not the input)
        predicted_words = []
        for seq in output:
            # Get only the newly generated part (after input length)
            new_tokens = seq[input_ids.shape[-1]:]
            word = gpt2_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Clean up the prediction (remove extra spaces, punctuation issues)
            word = word.split()[0] if word else ""
            word = word.strip('.,!?;:"')  # Remove surrounding punctuation
            
            if word:  # Only add if we got a valid word
                predicted_words.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in predicted_words:
            if word.lower() not in seen and word:
                seen.add(word.lower())
                unique_words.append(word)
        
        # Ensure we return exactly top_n predictions
        while len(unique_words) < top_n:
            unique_words.append("...")  # Fallback if not enough unique words
        
        return unique_words[:top_n]
    
    except Exception as e:
        print(f"Error in GPT-2 prediction: {str(e)}")
        return ["error"] * top_n

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        phrase = data.get("phrase", "").strip()
        model_type = data.get("model", "lstm").lower()

        print(f"Model selected: {model_type}")
        print(f"Phrase received: {phrase}")

        if model_type in models:
            model = models[model_type]
            predicted_words = predict_next_words(model, phrase)
        else:
            raise ValueError("Invalid model type")

        print(f"Predictions: {predicted_words}")
        return jsonify({"input": phrase, "predictions": predicted_words})

    except ValueError as e:
        print("Validation Error:", e)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/predict_gpt2", methods=["POST"])
def gpt2_predict():
    try:
        data = request.json
        phrase = data.get("phrase", "").strip()

        print(f"GPT-2 Phrase received: '{phrase}'")

        predicted_words = predict_gpt2(phrase)
        print(f"GPT-2 Predictions: {predicted_words}")
        return jsonify({"input": phrase, "predictions": predicted_words})

    except Exception as e:
        print("GPT-2 Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)