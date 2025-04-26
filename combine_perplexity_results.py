import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle


def load_rnn_perplexities():
    """
    Try to load perplexity values from saved files or compute them if available.
    Returns a dictionary of model name to perplexity value.
    """
    perplexities = {}

    # Try to load from existing perplexity results file
    if os.path.exists("rnn_perplexity_results.json"):
        with open("rnn_perplexity_results.json", "r") as f:
            perplexities = json.load(f)
            print(f"Loaded RNN perplexities from rnn_perplexity_results.json")
            return perplexities

    print("No saved RNN perplexity results found. Computing from notebook variables...")

    try:
        # Try to access variables from the notebook environment
        # This will only work if run within the notebook after perplexity calculation
        import sys

        # Look for variables in globals()
        if "lstm_perplexity" in globals():
            perplexities["LSTM"] = globals()["lstm_perplexity"]
            perplexities["BiLSTM"] = globals()["bilstm_perplexity"]
            perplexities["GRU"] = globals()["gru_perplexity"]
            perplexities["BiLSTM-GRU"] = globals()["bilstm_gru_perplexity"]
            print("Found perplexity values in globals()")
        else:
            print("Perplexity values not found in globals()")

            # Provide sample values if we can't find the actual ones
            perplexities = {"LSTM": 0, "BiLSTM": 0, "GRU": 0, "BiLSTM-GRU": 0}
            print(
                "Set placeholder values. Please run the notebook cells first to calculate perplexity."
            )

    except Exception as e:
        print(f"Error accessing notebook variables: {e}")

    return perplexities


def load_transformer_perplexities():
    """
    Load transformer model perplexity values from the JSON file.
    """
    perplexities = {}

    if os.path.exists("transformer_perplexity_results.json"):
        with open("transformer_perplexity_results.json", "r") as f:
            perplexities = json.load(f)
            print(
                f"Loaded transformer perplexities from transformer_perplexity_results.json"
            )
    else:
        print(
            "transformer_perplexity_results.json not found. Please run calculate_transformer_perplexity.py first."
        )

    return perplexities


def save_combined_perplexities(perplexities):
    """
    Save the combined perplexity results to a JSON file
    """
    with open("all_model_perplexity_results.json", "w") as f:
        json.dump(perplexities, f, indent=4)
    print(f"Combined perplexity results saved to all_model_perplexity_results.json")


def plot_perplexity_comparison(perplexities):
    """
    Generate a bar chart comparing perplexity across all models
    """
    models = list(perplexities.keys())
    values = list(perplexities.values())

    # Define colors for different model types
    colors = []
    for model in models:
        if any(rnn_type in model for rnn_type in ["LSTM", "GRU"]):
            colors.append("blue")  # RNN models in blue
        else:
            colors.append("green")  # Transformer models in green

    # Sort models by perplexity (lower is better)
    sorted_indices = np.argsort(values)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(sorted_models, sorted_values, color=sorted_colors)

    # Add perplexity values to the end of each bar
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{sorted_values[i]:.2f}",
            va="center",
        )

    plt.title("Model Comparison: Perplexity (Lower is Better)", fontsize=16)
    plt.xlabel("Perplexity", fontsize=14)
    plt.ylabel("Model", fontsize=14)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="blue", label="RNN-based Models"),
        Patch(facecolor="green", label="Transformer Models"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig("model_perplexity_comparison.png", dpi=300)
    plt.show()

    print(f"Perplexity comparison chart saved to model_perplexity_comparison.png")


def extract_perplexity_from_notebook():
    """
    Create a separate function to extract perplexity values directly from the notebook by running computation
    """
    try:
        import numpy as np
        from tensorflow.keras.models import load_model
        import math

        print("Loading models...")
        lstm_model = load_model("hinglish_lstm_model.keras")
        bilstm_model = load_model("hinglish_bilstm_model.keras")
        gru_model = load_model("hinglish_gru_model.keras")
        bilstm_gru_model = load_model("hinglish_bilstm_model.keras")

        # Load validation data
        X_val = np.load("X.npy")
        y_val = np.load("y.npy")

        # Use train_test_split to get validation set if not already split
        if not os.path.exists("X_val.npy"):
            from sklearn.model_selection import train_test_split

            _, X_val, _, y_val = train_test_split(
                X_val, y_val, test_size=0.2, random_state=42
            )

        def calculate_perplexity(model, X_data, y_data, batch_size=64):
            # Get the prediction probabilities
            y_pred_probs = model.predict(
                X_data[:5000], batch_size=batch_size
            )  # Limit to 5000 samples for memory

            # Extract the probabilities for the actual next words
            perplexities = []
            for i, y_true in enumerate(y_data[:5000]):
                # Get the predicted probability for the true next word
                true_prob = y_pred_probs[i][y_true]

                # Calculate perplexity: 2^(-log_2(p))
                if true_prob > 0:  # Avoid log(0)
                    perplexity = 2 ** (-math.log2(true_prob))
                    perplexities.append(perplexity)

            # Return the average perplexity
            return np.mean(perplexities)

        print("Calculating perplexities...")
        perplexities = {
            "LSTM": calculate_perplexity(lstm_model, X_val, y_val),
            "BiLSTM": calculate_perplexity(bilstm_model, X_val, y_val),
            "GRU": calculate_perplexity(gru_model, X_val, y_val),
            "BiLSTM-GRU": calculate_perplexity(bilstm_gru_model, X_val, y_val),
        }

        # Save the results
        with open("rnn_perplexity_results.json", "w") as f:
            json.dump(perplexities, f, indent=4)

        print(f"RNN perplexities calculated and saved to rnn_perplexity_results.json")
        return perplexities

    except Exception as e:
        print(f"Error computing perplexity: {e}")
        return {}


def main():
    print("Starting perplexity comparison...")

    # Try to load RNN perplexities
    rnn_perplexities = load_rnn_perplexities()

    # If couldn't load from file or globals, try to compute them
    if all(value == 0 for value in rnn_perplexities.values()):
        print("Computing perplexity values directly...")
        rnn_perplexities = extract_perplexity_from_notebook()

    # Load transformer perplexities
    transformer_perplexities = load_transformer_perplexities()

    # Combine the results
    all_perplexities = {**rnn_perplexities, **transformer_perplexities}

    if not all_perplexities:
        print(
            "No perplexity data found. Please run the notebook cells and transformer script first."
        )
        return

    # Save the combined results
    save_combined_perplexities(all_perplexities)

    # Plot the comparison
    plot_perplexity_comparison(all_perplexities)


if __name__ == "__main__":
    main()
