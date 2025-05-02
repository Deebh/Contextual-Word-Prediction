import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import math

# --- Load Models ---

model_dir = "rnn_models"
models = {}  # Dictionary to hold the loaded models
model_files = {
    "lstm": "hinglish_lstm_model.keras",  # Assuming this corresponds to lstm_model
    "bilstm": "hinglish_bilstm_model.keras",  # Assuming this corresponds to bilstm_model
    "gru": "gru_model.keras",  # Matches gru_model
    "bilstm_gru": "bilstm_gru_model.keras",  # Matches bilstm_gru_model
}

print("Loading models...")

for name, filename in model_files.items():
    model_path = os.path.join(model_dir, filename)
    variable_name = f"{name}_model"  # e.g., lstm_model, gru_model
    print(f"Attempting to load {variable_name} from {model_path}...")
    if os.path.exists(model_path):
        try:
            # Use a temporary variable, then assign to globals() or locals()
            # Using globals() makes them accessible outside this specific loop easily
            # Note: Modifying globals() directly is generally discouraged in complex code,
            # but acceptable here for setting up specific variables needed later.
            globals()[variable_name] = load_model(model_path)
            print(f"✅ Successfully loaded {variable_name}")
            # Optional: Print model summary
            # globals()[variable_name].summary()
        except Exception as e:
            print(f"❌ Error loading {variable_name} from {model_path}: {e}")
            globals()[variable_name] = None  # Set to None if loading failed
    else:
        print(f"⚠️ Model file not found: {model_path}. Cannot load {variable_name}.")
        globals()[variable_name] = None

# --- Check if all required models were loaded ---
required_models = ["lstm_model", "bilstm_model", "gru_model", "bilstm_gru_model"]
all_loaded = True
for model_var in required_models:
    if model_var not in globals() or globals()[model_var] is None:
        print(
            f"Error: Model variable '{model_var}' was not loaded successfully. Cannot proceed with perplexity calculation."
        )
        all_loaded = False

# Proceed only if all models are loaded
if all_loaded:
    print("\nAll required models loaded. Proceeding with perplexity calculation.")

    # -------------------------------------------------------------
    # Your existing Perplexity Calculation Code starts here
    # Ensure X_train, y_train, X_unseen, y_unseen are defined!
    # -------------------------------------------------------------

    # Function to calculate perplexity for a given model
    def calculate_perplexity(model, X_data, y_data, batch_size=64):
        if model is None:  # Check if model loaded correctly
            print("Skipping perplexity calculation for a model that failed to load.")
            return float("inf")  # Return infinity or handle appropriately

        print(
            f"  Calculating perplexity for model: {model.name}..."
        )  # model.name might be generic
        # get the full probability matrix: shape (num_samples, vocab_size)
        y_pred_probs = model.predict(
            X_data, batch_size=batch_size, verbose=0
        )  # Set verbose=0 for cleaner output

        # collect negative log‑probs
        neg_log_probs = []
        # Use np arrays for faster indexing if y_data is large
        y_data_arr = np.array(y_data)
        num_samples = len(y_data_arr)

        # Efficiently get the probabilities corresponding to the true next words
        # Use range(num_samples) for row indices and y_data_arr for column indices
        true_probs = y_pred_probs[
            np.arange(num_samples), y_data_arr.flatten()
        ]  # Ensure y_data is flat

        # Filter out zero probabilities and calculate negative log probabilities
        # Add a small epsilon to prevent log(0)
        epsilon = 1e-12
        valid_probs = true_probs[true_probs > epsilon]
        if len(valid_probs) < num_samples:
            print(
                f"  Warning: {num_samples - len(valid_probs)} zero probabilities encountered for true tokens."
            )
            if len(valid_probs) == 0:
                print(
                    "  Error: All true token probabilities were zero or less. Cannot calculate perplexity."
                )
                return float("inf")

        neg_log_probs = -np.log(valid_probs)  # Use natural log

        # average negative log‐prob
        avg_neg_log_prob = np.mean(neg_log_probs)
        # exponentiate once -> perplexity
        perplexity = math.exp(avg_neg_log_prob)
        print(f"  Finished calculation. Perplexity = {perplexity:.4f}")
        return perplexity

    # Calculate perplexity on a subset of training data (for reference)
    # !!! IMPORTANT: Make sure X_train, y_train, X_unseen, y_unseen ARE DEFINED before this point !!!
    try:
        subset_size = min(10000, len(X_train))  # Use at most 10,000 samples
        X_train_subset = X_train[:subset_size]
        y_train_subset = y_train[:subset_size]

        # Calculate perplexity on the unseen validation set as well
        unseen_subset_size = min(10000, len(X_unseen))
        X_unseen_subset = X_unseen[:unseen_subset_size]
        y_unseen_subset = y_unseen[:unseen_subset_size]
    except NameError as e:
        print(f"\n❌ Error: Data variable not defined ({e}).")
        print(
            "Please ensure X_train, y_train, X_unseen, y_unseen are loaded before calculating perplexity."
        )
        # Exit or prevent further execution
        exit()  # Or raise an exception

    print("\nComputing perplexity for all models...")
    print(
        f"Using subset sizes: Train={len(X_train_subset)}, Unseen={len(X_unseen_subset)}"
    )
    print("This may take some time depending on the subset size and model complexity.")

    # --- Calculate Training Perplexity ---
    print("\n--- Training Set Perplexity ---")
    lstm_perplexity = calculate_perplexity(lstm_model, X_train_subset, y_train_subset)
    print(f"LSTM Model Training Perplexity: {lstm_perplexity:.4f}")

    bilstm_perplexity = calculate_perplexity(
        bilstm_model, X_train_subset, y_train_subset
    )
    print(f"BiLSTM Model Training Perplexity: {bilstm_perplexity:.4f}")

    gru_perplexity = calculate_perplexity(gru_model, X_train_subset, y_train_subset)
    print(f"GRU Model Training Perplexity: {gru_perplexity:.4f}")

    bilstm_gru_perplexity = calculate_perplexity(
        bilstm_gru_model, X_train_subset, y_train_subset
    )
    print(f"BiLSTM-GRU Model Training Perplexity: {bilstm_gru_perplexity:.4f}")

    # --- Calculate Unseen Perplexity ---
    print("\n--- Unseen Validation Set Perplexity ---")
    lstm_unseen_perplexity = calculate_perplexity(
        lstm_model, X_unseen_subset, y_unseen_subset
    )
    print(f"LSTM Model Unseen Perplexity: {lstm_unseen_perplexity:.4f}")

    bilstm_unseen_perplexity = calculate_perplexity(
        bilstm_model, X_unseen_subset, y_unseen_subset
    )
    print(f"BiLSTM Model Unseen Perplexity: {bilstm_unseen_perplexity:.4f}")

    gru_unseen_perplexity = calculate_perplexity(
        gru_model, X_unseen_subset, y_unseen_subset
    )
    print(f"GRU Model Unseen Perplexity: {gru_unseen_perplexity:.4f}")

    bilstm_gru_unseen_perplexity = calculate_perplexity(
        bilstm_gru_model, X_unseen_subset, y_unseen_subset
    )
    print(f"BiLSTM-GRU Model Unseen Perplexity: {bilstm_gru_unseen_perplexity:.4f}")

else:
    print("\nSkipping perplexity calculation due to model loading errors.")
