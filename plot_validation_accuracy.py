import json
import matplotlib.pyplot as plt

# Load the history from JSON file
with open("hinglish_model_history.json", "r") as f:
    loaded_history = json.load(f)

# Plot validation accuracy for all models (assuming the JSON saved lists of accuracies)
plt.figure(figsize=(10, 6))
plt.plot(loaded_history["LSTM"]["val_accuracy"], label="LSTM")
plt.plot(loaded_history["BiLSTM"]["val_accuracy"], label="BiLSTM")
plt.plot(loaded_history["GRU"]["val_accuracy"], label="GRU")
plt.plot(loaded_history["BiLSTM-GRU"]["val_accuracy"], label="BiLSTM-GRU")
plt.title("Model Comparison: Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()
