import torch
from transformers import XLNetTokenizer, XLNetLMHeadModel, pipeline


def generate_text(
    prompt, model_path="./xlnet-hinglish-final", max_length=100, min_length=10
):
    """
    Generate text using a fine-tuned XLNet model with better generation parameters.

    Args:
        prompt (str): The input text to generate from
        model_path (str): Path to the fine-tuned model
        max_length (int): Maximum length of generated text
        min_length (int): Minimum length of generated text

    Returns:
        list: Generated text sequences
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device set to use {device}")

    # Load model and tokenizer
    try:
        model = XLNetLMHeadModel.from_pretrained(model_path).to(device)
        tokenizer = XLNetTokenizer.from_pretrained(model_path)
        print(f"Model and tokenizer loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

    # Handle pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    # Create pipeline with specific parameters
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    # Generate text with careful parameters
    try:
        print(f"Generating text from prompt: '{prompt}'")
        output = generator(
            prompt,
            max_length=max_length,
            min_length=min_length,
            num_return_sequences=3,  # Generate multiple sequences
            do_sample=True,  # Use sampling
            top_k=50,  # Sample from top 50 tokens
            top_p=0.9,  # Nucleus sampling
            temperature=0.8,  # Higher temperature for more randomness
            no_repeat_ngram_size=2,  # Avoid repeating 2-grams
            early_stopping=False,  # Don't stop early
        )
        return output
    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        return None


if __name__ == "__main__":
    print("XLNet Hinglish Text Generator")
    print("----------------------------")

    while True:
        user_input = input("\nEnter your prompt (or 'quit' to exit): ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        results = generate_text(user_input)

        if results:
            print("\n=== Generated Text ===")
            for i, result in enumerate(results):
                print(f"\nOption {i+1}:")
                print(result["generated_text"])
            print("=====================")
