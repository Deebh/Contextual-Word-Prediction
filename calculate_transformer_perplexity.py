import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import json
import os
import re


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_perplexity_transformer(model, tokenizer, texts, max_length=128):
    """
    Calculate perplexity for transformer models like GPT-2 and XLNet
    """
    model.eval()
    total_nlls = []
    total_tokens = 0

    # Process texts in batches to avoid memory issues
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_texts = [preprocess_text(text) for text in batch_texts]

        # Tokenize input
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        model = model.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )

            # Calculate negative log likelihood
            nll = (
                outputs.loss.item() * input_ids.shape[0]
            )  # Multiply by batch size as the loss is averaged
            total_nlls.append(nll)
            total_tokens += torch.sum(attention_mask).item()

    avg_nll = sum(total_nlls) / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_nll)
    return perplexity


def main():
    print("Loading dataset...")
    try:
        df = pd.read_csv("final_dataset.csv")
        # Use a subset of the data for perplexity calculation to avoid memory issues
        sample_size = min(1000, len(df))
        texts = df["Conversation"].astype(str).values[:sample_size]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        texts = [
            "This is a fallback text to ensure the script runs even without proper data"
        ]

    results = {}

    # List all transformer models to evaluate
    models_to_evaluate = [
        {
            "name": "GPT-2 Hinglish Fine-tuned",
            "path": "gpt2_hinglish_model_final",
            "model_type": "gpt2",
        },
        {
            "name": "GPT-2 Hinglish (Checkpoint 3125)",
            "path": "gpt2_hinglish_finetuned/checkpoint-3125",
            "model_type": "gpt2",
        },
    ]

    # Add XLNet if there's a saved model
    if os.path.exists("xlnet_hinglish_model"):
        models_to_evaluate.append(
            {
                "name": "XLNet Hinglish",
                "path": "xlnet_hinglish_model",
                "model_type": "xlnet",
            }
        )

    # Calculate perplexity for each model
    for model_info in models_to_evaluate:
        try:
            print(f"Calculating perplexity for {model_info['name']}...")

            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_info["path"])
            tokenizer = AutoTokenizer.from_pretrained(model_info["path"])

            # Calculate perplexity
            perplexity = calculate_perplexity_transformer(model, tokenizer, texts)

            print(f"{model_info['name']} perplexity: {perplexity:.4f}")
            results[model_info["name"]] = perplexity

        except Exception as e:
            print(f"Error calculating perplexity for {model_info['name']}: {e}")

    # Save results to a JSON file
    with open("transformer_perplexity_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to transformer_perplexity_results.json")


if __name__ == "__main__":
    main()
