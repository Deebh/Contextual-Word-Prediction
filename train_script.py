# %%
import pandas as pd
import numpy as np
import re
import gc  # Garbage collector
import math
import json
import os
import time  # To time operations

# Import TensorFlow and Keras components
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import mixed_precision  # Import mixed precision
import matplotlib.pyplot as plt

# %%
# --- Enable Mixed Precision ---
# Use float16 computations where possible for speed and VRAM savings
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)
print(
    "Mixed precision enabled: Compute dtype=%s, Variable dtype=%s"
    % (policy.compute_dtype, policy.variable_dtype)
)

# Check for GPU availability
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Set memory growth to avoid allocating all VRAM at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# %%
# --- Configuration ---
FILE_PATH = "final_dataset.csv"
# Adjusted for 24GB RAM - monitor your usage! Reduce if needed.
CHUNK_SIZE = 50000
MAX_SEQUENCE_LENGTH = 100  # Keep moderate, reducing helps memory but might hurt context
EMBEDDING_DIM = 100
LSTM_UNITS = 128  # Keep units moderate to balance VRAM/performance
# Increased batch size - *TUNE THIS* based on your VRAM. Start high, reduce if errors.
BATCH_SIZE_FIT = 128  # Try 128, if OOM error -> 96, 64, or 32
TOTAL_EPOCHS = 6  # Increased epochs for potentially better accuracy


# --- Preprocessing Function ---
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)  # Ensure text is string
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# %%
# --- Pass 1: Build Tokenizer Vocabulary ---
print("--- Starting Pass 1: Building Tokenizer ---")
tokenizer_start_time = time.time()
tokenizer = Tokenizer(oov_token="<OOV>")  # Add Out-of-Vocabulary token
row_count = 0
try:
    for i, chunk in enumerate(pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE)):
        print(f"Processing chunk {i+1} for tokenizer...")
        if "Conversation" not in chunk.columns:
            print(f"Warning: 'Conversation' column not found in chunk {i+1}. Skipping.")
            continue
        # Process in place to potentially save memory if pandas optimizes
        chunk["Conversation"] = chunk["Conversation"].apply(preprocess_text)
        tokenizer.fit_on_texts(chunk["Conversation"])
        row_count += len(chunk)
        del chunk  # Explicit cleanup
        gc.collect()
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()
except Exception as e:
    print(f"An error occurred during Pass 1: {e}")
    exit()

vocab_size = len(tokenizer.word_index) + 1
tokenizer_end_time = time.time()
print(
    f"--- Pass 1 Complete ({(tokenizer_end_time - tokenizer_start_time):.2f} seconds) ---"
)
print(f"Total rows processed: {row_count}")
print(f"Vocabulary Size: {vocab_size}")

# Save the tokenizer
tokenizer_json = tokenizer.to_json()
with open("hinglish_tokenizer.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
print("Tokenizer saved to hinglish_tokenizer.json")
gc.collect()


# %%
# --- Function to Prepare Data Chunk ---
def prepare_data_chunk(chunk_df, tokenizer, max_sequence_length):
    """Preprocesses, tokenizes, and creates sequences for a data chunk."""
    prep_start_time = time.time()
    if "Conversation" not in chunk_df.columns:
        print("Warning: 'Conversation' column missing in current chunk.")
        return None, None

    # Assume preprocessing was done inplace during tokenization pass or apply again if needed
    # chunk_df["Conversation_Processed"] = chunk_df["Conversation"].apply(preprocess_text) # Optional: re-apply if not done inplace
    sequences = tokenizer.texts_to_sequences(
        chunk_df["Conversation"]
    )  # Use original if preprocessed inplace

    X_chunk, y_chunk = [], []
    for seq in sequences:
        # Limit sequence generation length per conversation
        for i in range(1, min(len(seq), max_sequence_length + 1)):
            start_index = max(0, i - max_sequence_length)
            input_seq = seq[start_index:i]
            output_word = seq[i]
            X_chunk.append(input_seq)
            y_chunk.append(output_word)

    if not X_chunk:
        print("Warning: No sequences generated for this chunk.")
        del chunk_df, sequences  # Cleanup even if no sequences
        gc.collect()
        return None, None

    # Padding
    pad_start_time = time.time()
    X_chunk_padded = pad_sequences(X_chunk, maxlen=max_sequence_length, padding="pre")
    y_chunk_array = np.array(y_chunk)
    pad_end_time = time.time()

    prep_end_time = time.time()
    # print(f"    Chunk Prep Time: {(prep_end_time - prep_start_time):.2f}s (Padding: {(pad_end_time - pad_start_time):.2f}s)")

    # Clean up intermediate data for this chunk
    del chunk_df, sequences, X_chunk, y_chunk
    gc.collect()

    return X_chunk_padded, y_chunk_array


# %%
# --- Define Models (with mixed precision considerations) ---


# Helper function to build models
def build_model(model_type, vocab_size, embedding_dim, rnn_units, max_sequence_length):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_sequence_length,
        )
    )

    if model_type == "lstm":
        model.add(LSTM(units=rnn_units, return_sequences=False))
    elif model_type == "bilstm":
        model.add(Bidirectional(LSTM(units=rnn_units, return_sequences=False)))
    elif model_type == "gru":
        model.add(GRU(units=rnn_units, return_sequences=False))
    elif model_type == "bilstm-gru":
        model.add(Bidirectional(LSTM(units=rnn_units, return_sequences=True)))
        model.add(GRU(units=rnn_units, return_sequences=False))
    else:
        raise ValueError("Unknown model type")

    # Add the final dense layer - crucial to set dtype for mixed precision stability
    model.add(
        Dense(units=vocab_size, activation="softmax", dtype="float32")
    )  # Use float32 for output

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


# Instantiate models
lstm_model = build_model(
    "lstm", vocab_size, EMBEDDING_DIM, LSTM_UNITS, MAX_SEQUENCE_LENGTH
)
bilstm_model = build_model(
    "bilstm", vocab_size, EMBEDDING_DIM, LSTM_UNITS, MAX_SEQUENCE_LENGTH
)
gru_model = build_model(
    "gru", vocab_size, EMBEDDING_DIM, LSTM_UNITS, MAX_SEQUENCE_LENGTH
)
bilstm_gru_model = build_model(
    "bilstm-gru", vocab_size, EMBEDDING_DIM, LSTM_UNITS, MAX_SEQUENCE_LENGTH
)

models = {
    "LSTM": lstm_model,
    "BiLSTM": bilstm_model,
    "GRU": gru_model,
    "BiLSTM-GRU": bilstm_gru_model,
}

model_histories = {name: {"accuracy": [], "loss": []} for name in models.keys()}

print("--- Models Defined ---")
lstm_model.summary()  # Display summary for one model example

# %%
# --- Pass 2: Training Loop ---
overall_training_start_time = time.time()

for model_name, model in models.items():
    print(f"\n--- Training Model: {model_name} ---")
    model_start_time = time.time()
    current_history = model_histories[model_name]

    for epoch in range(TOTAL_EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{TOTAL_EPOCHS} for {model_name}")
        epoch_loss = []
        epoch_accuracy = []
        chunk_num = 0
        total_samples_in_epoch = 0

        try:
            # Read data in chunks for training
            chunk_iterator = pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE)
            for chunk in chunk_iterator:
                chunk_num += 1
                print(f"  Epoch {epoch+1}, Chunk {chunk_num}: Preparing data...")

                X_chunk, y_chunk = prepare_data_chunk(
                    chunk, tokenizer, MAX_SEQUENCE_LENGTH
                )

                if X_chunk is None or y_chunk is None or len(X_chunk) == 0:
                    print(f"  Skipping empty or invalid chunk {chunk_num}.")
                    del chunk  # Explicitly delete chunk dataframe
                    gc.collect()
                    continue

                num_samples = len(X_chunk)
                total_samples_in_epoch += num_samples
                print(
                    f"  Epoch {epoch+1}, Chunk {chunk_num}: Training on {num_samples} samples (Batch Size: {BATCH_SIZE_FIT})..."
                )

                # Use tf.data for potential prefetching (optional but can help)
                # train_dataset = tf.data.Dataset.from_tensor_slices((X_chunk, y_chunk))
                # train_dataset = train_dataset.batch(BATCH_SIZE_FIT).prefetch(tf.data.AUTOTUNE)
                # history = model.fit(train_dataset, epochs=1, verbose=1)
                # Simpler fit:
                history = model.fit(
                    X_chunk, y_chunk, epochs=1, batch_size=BATCH_SIZE_FIT, verbose=1
                )

                # Record metrics
                if history.history:
                    # Weight chunk metrics by number of samples if averaging later
                    epoch_loss.extend(
                        history.history["loss"]
                    )  # Store raw losses per batch
                    epoch_accuracy.extend(
                        history.history["accuracy"]
                    )  # Store raw acc per batch

                # Explicitly free memory
                del (
                    X_chunk,
                    y_chunk,
                    chunk,
                    history,
                )  # , train_dataset (if using tf.data)
                gc.collect()

        except FileNotFoundError:
            print(f"Error: File not found at {FILE_PATH} during training.")
            break
        except tf.errors.ResourceExhaustedError as e:
            print(f"\n\n\n ****** VRAM EXHAUSTED ****** ")
            print(f"Error during training chunk {chunk_num} for {model_name}: {e}")
            print(
                f"Try reducing BATCH_SIZE_FIT (currently {BATCH_SIZE_FIT}). Also check CHUNK_SIZE & MAX_SEQUENCE_LENGTH."
            )
            print("Stopping training for this model.")
            # Mark history as potentially incomplete or invalid
            current_history["loss"].append(float("nan"))
            current_history["accuracy"].append(float("nan"))
            gc.collect()  # Try to free VRAM
            break  # Stop training this model
        except Exception as e:
            print(
                f"An unexpected error occurred during training chunk {chunk_num} for {model_name}: {e}"
            )
            break  # Stop processing this epoch

        # Check if ResourceExhaustedError occurred and broke the inner loop
        if (
            np.isnan(current_history["loss"]).any()
            or np.isnan(current_history["accuracy"]).any()
        ):
            print(
                f"Skipping end-of-epoch summary due to previous error for {model_name}."
            )
            break  # Stop training this model entirely if one epoch failed catastrophically

        # Store average metrics for the completed epoch
        avg_epoch_loss = np.mean(epoch_loss) if epoch_loss else 0
        avg_epoch_accuracy = np.mean(epoch_accuracy) if epoch_accuracy else 0
        current_history["loss"].append(avg_epoch_loss)
        current_history["accuracy"].append(avg_epoch_accuracy)
        epoch_end_time = time.time()

        print(
            f"Epoch {epoch + 1} summary for {model_name}: Avg Loss: {avg_epoch_loss:.4f}, Avg Accuracy: {avg_epoch_accuracy:.4f}"
        )
        print(
            f"Epoch {epoch + 1} took {(epoch_end_time - epoch_start_time):.2f} seconds for {total_samples_in_epoch} samples."
        )
        gc.collect()  # Clean up at end of epoch

    # --- Final Model Saving ---
    model_end_time = time.time()
    print(
        f"--- Training Complete for {model_name} ({(model_end_time - model_start_time):.2f} seconds) ---"
    )
    # Only save if training didn't hit resource exhaustion
    if not (
        np.isnan(current_history["loss"]).any()
        or np.isnan(current_history["accuracy"]).any()
    ):
        model_filename = (
            f"hinglish_{model_name.lower()}_model_chunked_ep{TOTAL_EPOCHS}.keras"
        )
        model.save(model_filename)
        print(f"{model_name} model saved to {model_filename}")
    else:
        print(
            f"Skipping saving {model_name} model due to training errors (Resource Exhausted?)."
        )

    gc.collect()  # Clean up before next model

overall_training_end_time = time.time()
print(
    f"\n--- Total Training Time for all models: {(overall_training_end_time - overall_training_start_time):.2f} seconds ---"
)

# %%
# --- Save Training Histories ---
history_filename = f"hinglish_model_history_chunked_ep{TOTAL_EPOCHS}.json"
serializable_history = {}
for model_name, hist in model_histories.items():
    # Ensure lists aren't empty and convert numpy floats/NaNs
    serializable_history[model_name] = {
        k: [float(val) if not np.isnan(val) else "NaN" for val in v] if v else []
        for k, v in hist.items()
    }

try:
    with open(history_filename, "w") as f:
        json.dump(serializable_history, f, indent=4)
    print(f"✅ All model training histories saved to {history_filename}")
except Exception as e:
    print(f"Error saving history: {e}")

# %%
# --- Plot Training Accuracy (Example) ---
plt.figure(figsize=(12, 6))
for model_name, history in model_histories.items():
    # Filter out NaN values before plotting
    acc = [
        val
        for val in history.get("accuracy", [])
        if val is not None and not np.isnan(val)
    ]
    if acc:
        plt.plot(acc, label=f"{model_name} Accuracy")

plt.title(
    f"Model Training Accuracy per Epoch (Chunked Training, {TOTAL_EPOCHS} Epochs)"
)
plt.xlabel("Epochs")
plt.ylabel("Average Training Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# %%
# --- Prediction Functions ---
# Load tokenizer (already loaded usually, but good practice if running cells separately)
try:
    with open("hinglish_tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_json_loaded = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
            tokenizer_json_loaded
        )
    print("Tokenizer loaded successfully.")
except FileNotFoundError:
    print("Error: hinglish_tokenizer.json not found. Cannot run predictions.")
    exit()
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()


# Function to load a specific model
def load_specific_model(model_name, epoch_count):
    model_filename = (
        f"hinglish_{model_name.lower()}_model_chunked_ep{epoch_count}.keras"
    )
    if not os.path.exists(model_filename):
        print(f"Model file not found: {model_filename}. Was training successful?")
        return None
    try:
        # When loading a model saved with mixed precision, it should handle it automatically
        model = load_model(model_filename)
        print(f"{model_name} model loaded successfully from {model_filename}.")
        return model
    except Exception as e:
        print(f"Error loading model {model_filename}: {e}")
        # Attempt loading without custom objects if error is related (less likely here)
        # try:
        #     model = load_model(model_filename, compile=False)
        #     model.compile(...) # Recompile manually
        # except: ...
        return None


# Load one model for prediction examples (e.g., LSTM)
loaded_model_name = "lstm"  # Change as needed
loaded_model = load_specific_model(loaded_model_name, TOTAL_EPOCHS)


def predict_top_words(input_text, tokenizer, model, max_sequence_length, top_n=3):
    if model is None:
        print("Model not loaded. Cannot predict.")
        return []
    input_text = preprocess_text(input_text)
    try:
        sequence = tokenizer.texts_to_sequences([input_text])[0]
    except Exception as e:
        print(f"Error tokenizing input: {e}. Input: '{input_text}'")
        return []
    if not sequence:
        print(
            f"Input text '{input_text}' resulted in empty sequence after tokenization."
        )
        return []

    sequence_padded = pad_sequences(
        [sequence], maxlen=max_sequence_length, padding="pre"
    )

    try:
        # Prediction might be faster with mixed precision enabled globally
        predict_start = time.time()
        predictions = model.predict(sequence_padded)[0]
        predict_end = time.time()
        # print(f"Prediction time: {(predict_end - predict_start)*1000:.2f} ms")

        # Ensure predictions are float32 before argsort for stability if needed
        # predictions = predictions.astype('float32')

        top_indices = np.argsort(predictions)[-top_n:][::-1]
        top_words = [
            tokenizer.index_word.get(idx, "<OOV>") for idx in top_indices
        ]  # Use OOV
        return top_words
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return []


# %%
# --- User Prompt for Prediction ---
if loaded_model:
    try:
        input_text_prompt = input(
            f"Enter a Hinglish phrase (using loaded {loaded_model_name.upper()} model): "
        )
        predicted_words = predict_top_words(
            input_text_prompt, tokenizer, loaded_model, MAX_SEQUENCE_LENGTH
        )
        print(f"Predicted next words: {predicted_words}")
    except EOFError:
        print("\nSkipping user input prompt in non-interactive mode.")
else:
    print("Skipping prediction prompt as model loading failed or was skipped.")


# %%
# --- Perplexity Calculation (Remains largely the same, but uses updated model names) ---

VALIDATION_SET_SIZE = 10000  # Keep perplexity calculation manageable
print(
    f"\n--- Calculating Perplexity (using first {VALIDATION_SET_SIZE} rows as pseudo-validation set) ---"
)

X_val_list, y_val_list = [], []
rows_collected = 0
try:
    val_load_start = time.time()
    # Use smaller chunksize for validation loading to avoid RAM spikes if VAL_SET_SIZE is large
    val_chunk_size = min(CHUNK_SIZE // 2, VALIDATION_SET_SIZE)
    for i, chunk in enumerate(pd.read_csv(FILE_PATH, chunksize=val_chunk_size)):
        # print(f"  Loading chunk {i+1} for validation data...")
        if "Conversation" not in chunk.columns:
            continue

        # Preprocess - use the already loaded function
        chunk["Conversation"] = chunk["Conversation"].apply(preprocess_text)
        sequences = tokenizer.texts_to_sequences(chunk["Conversation"])

        for seq in sequences:
            for j in range(1, min(len(seq), MAX_SEQUENCE_LENGTH + 1)):
                if rows_collected >= VALIDATION_SET_SIZE:
                    break
                start_index = max(0, j - MAX_SEQUENCE_LENGTH)
                input_seq = seq[start_index:j]
                output_word = seq[j]
                # Simple check if word is in vocab before adding
                if output_word < vocab_size:  # Index should be less than vocab size
                    X_val_list.append(input_seq)
                    y_val_list.append(output_word)
                    rows_collected += 1
            if rows_collected >= VALIDATION_SET_SIZE:
                break
        if rows_collected >= VALIDATION_SET_SIZE:
            break
        del chunk, sequences
        gc.collect()
    val_load_end = time.time()

    if not X_val_list:
        print("Error: Could not collect any valid validation data.")
        X_val_subset, y_val_subset = None, None
    else:
        X_val_subset = pad_sequences(
            X_val_list, maxlen=MAX_SEQUENCE_LENGTH, padding="pre"
        )
        y_val_subset = np.array(y_val_list)
        print(
            f"Collected {len(X_val_subset)} samples for perplexity in {(val_load_end - val_load_start):.2f}s."
        )
        del X_val_list, y_val_list
        gc.collect()

except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH} while loading validation data.")
    X_val_subset, y_val_subset = None, None
except Exception as e:
    print(f"An error occurred loading validation data: {e}")
    X_val_subset, y_val_subset = None, None


# Perplexity function (minor improvements for stability)
def calculate_perplexity(model, X_data, y_data, batch_size=64, vocab_size_check=None):
    if X_data is None or y_data is None or len(X_data) == 0:
        print("Invalid data provided for perplexity calculation.")
        return float("inf")

    print(f"  Predicting probabilities for {len(X_data)} samples...")
    try:
        # Use a potentially larger batch size for inference if VRAM allows
        pred_batch_size = batch_size * 2
        y_pred_probs = model.predict(X_data, batch_size=pred_batch_size, verbose=1)
        # Ensure output is float32 for stability if model outputs float16
        if y_pred_probs.dtype != np.float32:
            y_pred_probs = y_pred_probs.astype(np.float32)

    except tf.errors.ResourceExhaustedError as e:
        print(
            f"  Resource exhausted during prediction for perplexity. Try reducing VALIDATION_SET_SIZE or batch_size. Error: {e}"
        )
        return float("inf")
    except Exception as e:
        print(f"  Error during prediction for perplexity: {e}")
        return float("inf")

    neg_log_probs = []
    print("  Calculating negative log probabilities...")
    num_skipped_oob = 0  # Out of bounds
    num_skipped_zero = 0  # Zero probability

    # Get actual model output layer size if not passed
    if vocab_size_check is None:
        vocab_size_check = model.output_shape[-1]

    for i, true_idx in enumerate(y_data):
        if true_idx >= vocab_size_check or true_idx < 0:
            num_skipped_oob += 1
            continue

        p = y_pred_probs[i][true_idx]

        if p <= 1e-9:  # Increased epsilon slightly
            num_skipped_zero += 1
            # Using a large NLL is better than skipping entirely sometimes
            neg_log_probs.append(30.0)  # High penalty
        else:
            neg_log_probs.append(-math.log(p))

    if num_skipped_oob > 0:
        print(
            f"  Skipped {num_skipped_oob} samples due to out-of-bounds true indices (True index >= {vocab_size_check})."
        )
    if num_skipped_zero > 0:
        print(
            f"  Used high penalty for {num_skipped_zero} samples with near-zero probability."
        )

    if not neg_log_probs:
        print("  No valid log probabilities calculated.")
        return float("inf")

    avg_neg_log_prob = np.mean(neg_log_probs)
    # Handle potential overflow if avg_neg_log_prob is huge
    try:
        perplexity = math.exp(
            min(avg_neg_log_prob, 700)
        )  # Cap input to exp to avoid overflow
    except OverflowError:
        perplexity = float("inf")
    return perplexity


# Calculate and store perplexities
perplexities = {}
if X_val_subset is not None and y_val_subset is not None:
    for model_name in models.keys():
        print(f"\nCalculating perplexity for {model_name}...")
        # Load the final saved model for evaluation
        model_to_eval = load_specific_model(model_name, TOTAL_EPOCHS)
        if model_to_eval:
            perp_start = time.time()
            perp = calculate_perplexity(
                model_to_eval,
                X_val_subset,
                y_val_subset,
                batch_size=BATCH_SIZE_FIT,  # Use training batch size or larger
                vocab_size_check=vocab_size,  # Pass the known vocab size
            )
            perp_end = time.time()
            print(
                f"{model_name} Model Perplexity: {perp:.4f} (calculated in {(perp_end-perp_start):.2f}s)"
            )
            perplexities[model_name] = perp
            del model_to_eval
            gc.collect()
        else:
            print(f"Skipping perplexity for {model_name} as model loading failed.")
            perplexities[model_name] = float("inf")
else:
    print("\nSkipping perplexity calculation due to issues loading validation data.")
    perplexities = {name: float("inf") for name in models.keys()}


# %%
# --- Save Perplexity Results ---
def save_perplexities(perplexity_dict):
    serializable_perplexities = {
        k: float(v) if not np.isinf(v) and not np.isnan(v) else "Infinity"
        for k, v in perplexity_dict.items()
    }
    filename = f"rnn_perplexity_results_chunked_ep{TOTAL_EPOCHS}.json"
    try:
        with open(filename, "w") as f:
            json.dump(serializable_perplexities, f, indent=4)
        print(f"Successfully saved perplexity values to {filename}")
    except Exception as e:
        print(f"Error saving perplexities to {filename}: {e}")
    return serializable_perplexities


saved_perplexities = save_perplexities(perplexities)

# %%
# --- Plot Perplexity Comparison ---
valid_models = [
    m for m, p in perplexities.items() if p != float("inf") and not np.isnan(p)
]
valid_perplexities = [perplexities[m] for m in valid_models]

if valid_models:
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_models)))  # Use a colormap
    bars = plt.bar(valid_models, valid_perplexities, color=colors)

    for bar, p in zip(bars, valid_perplexities):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval * 1.01,
            f"{p:.2f}",
            va="bottom",
            ha="center",
        )  # Adjust label position

    plt.title(f"Model Comparison: Perplexity (Lower is Better) - {TOTAL_EPOCHS} Epochs")
    plt.ylabel("Perplexity")
    if valid_perplexities:
        plt.ylim(0, max(valid_perplexities) * 1.2 + 5)  # Add a bit more space at top
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=15)  # Rotate labels slightly if they overlap
    plt.tight_layout()  # Adjust layout
    plt.show()
else:
    print("No valid perplexity values to plot.")

print("\n--- Script Finished ---")
