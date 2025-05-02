import os
import glob
import json
import matplotlib.pyplot as plt

# Load histories
histories = {}
history_files = glob.glob(
    "rnn_models/*_history.json"
)  # Use a more specific pattern if needed

if not history_files:
    print("Warning: No history files found matching 'rnn_models/*_history.json'.")
    # Exit or handle the case where no files are found
else:
    print(f"Found history files: {history_files}")
    for fp in history_files:
        name = os.path.basename(fp).replace("_history.json", "")
        try:
            with open(fp) as f:
                histories[name] = json.load(f)
        except Exception as e:
            print(f"Error loading {fp}: {e}")

# Check if any histories were successfully loaded
if not histories:
    print("No valid histories loaded. Cannot generate plots.")
else:
    # Calculate minimum lengths to avoid errors if histories have different lengths
    # Check if keys exist before accessing len
    min_epochs = 0
    if all(
        "accuracy" in h and isinstance(h["accuracy"], list) for h in histories.values()
    ):
        min_epochs = min(
            len(h["accuracy"])
            for h in histories.values()
            if "accuracy" in h and isinstance(h["accuracy"], list)
        )
    else:
        print("Warning: Not all histories contain a valid 'accuracy' list.")

    min_batches = 0
    if all(
        "unseen_accuracy" in h and isinstance(h["unseen_accuracy"], list)
        for h in histories.values()
    ):
        min_batches = min(
            len(h["unseen_accuracy"])
            for h in histories.values()
            if "unseen_accuracy" in h and isinstance(h["unseen_accuracy"], list)
        )
    else:
        print("Warning: Not all histories contain a valid 'unseen_accuracy' list.")

    # --- Plot 1: Training Accuracy ---

    # Increase default font sizes (apply globally)
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    # Create the first figure and axes
    fig1, ax1 = plt.subplots(
        figsize=(12, 7), dpi=120
    )  # Adjusted figsize for single plot

    if min_epochs > 0:
        # Plot training accuracy
        for name, h in histories.items():
            # Check if this specific history has the key and it's a list
            if "accuracy" in h and isinstance(h["accuracy"], list):
                ax1.plot(
                    h["accuracy"][:min_epochs],  # Use min_epochs for consistent x-axis
                    label=name,
                    linewidth=3,  # thicker lines
                )
            else:
                print(
                    f"Skipping 'accuracy' plot for {name} due to missing/invalid data."
                )

        ax1.set_title("Training Accuracy")
        ax1.set_xlabel(
            "Epoch"
        )  # Epochs correspond to the 'accuracy' and 'loss' keys from model.fit
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout(pad=1.5)  # Adjust layout for this figure
    else:
        print(
            "Skipping Training Accuracy plot as no valid data was found (min_epochs=0)."
        )

    # --- Plot 2: Unseen Validation Accuracy ---

    # Create the second figure and axes
    fig2, ax2 = plt.subplots(
        figsize=(12, 7), dpi=120
    )  # Adjusted figsize for single plot

    if min_batches > 0:
        # Plot unseen validation accuracy
        for name, h in histories.items():
            # Check if this specific history has the key and it's a list
            if "unseen_accuracy" in h and isinstance(h["unseen_accuracy"], list):
                ax2.plot(
                    h["unseen_accuracy"][
                        :min_batches
                    ],  # Use min_batches for consistent x-axis
                    label=name,
                    linewidth=3,
                )
            else:
                print(
                    f"Skipping 'unseen_accuracy' plot for {name} due to missing/invalid data."
                )

        ax2.set_title("Unseen Validation Accuracy")
        ax2.set_xlabel(
            "Batch Iteration"
        )  # Batches correspond to the 'unseen_accuracy' appended after each outer loop iteration
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout(pad=1.5)  # Adjust layout for this figure
    else:
        print(
            "Skipping Unseen Validation Accuracy plot as no valid data was found (min_batches=0)."
        )

    # Show both plots (will appear as separate windows or inline figures depending on environment)
    plt.show()
