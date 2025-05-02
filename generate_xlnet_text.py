import os
import torch
from transformers import XLNetTokenizer, XLNetLMHeadModel, set_seed


def generate_text_manual(
    prompt: str,
    model_path: str = "temp/xlnet-hinglish-chunk22",
    max_new_tokens: int = 100,
    num_return_sequences: int = 3,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 0.8,
):
    # device & reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # load model & tokenizer
    model_path = os.path.abspath(os.path.expanduser(model_path))
    model = XLNetLMHeadModel.from_pretrained(model_path, local_files_only=True).to(
        device
    )
    tokenizer = XLNetTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    seq_len = input_ids.size(-1)
    total_len = seq_len + max_new_tokens

    # build perm_mask: forbid new→future_new
    perm_mask = torch.zeros((1, total_len, total_len), dtype=torch.float, device=device)
    perm_mask[:, seq_len:, seq_len:] = 1.0

    # build target_mapping: predict only new token positions
    target_mapping = torch.zeros(
        (1, max_new_tokens, total_len), dtype=torch.float, device=device
    )
    for i in range(max_new_tokens):
        target_mapping[0, i, seq_len + i] = 1.0

    # generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=total_len,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # decode only the new tokens
    texts = [
        tokenizer.decode(output[seq_len:], skip_special_tokens=True)
        for output in outputs
    ]
    return texts


if __name__ == "__main__":
    print("XLNet Hinglish Text Generator (Manual Permutation)")
    while True:
        prompt = input("Enter prompt (or 'quit' to exit): ")
        if prompt.lower() in ("quit", "exit", "q"):
            break
        results = generate_text_manual(prompt)
        for idx, text in enumerate(results, start=1):
            print(f"\nOption {idx}:\n{text}")
        print("\n" + "=" * 40 + "\n")
