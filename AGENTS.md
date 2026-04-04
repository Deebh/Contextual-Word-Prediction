# AGENTS.md

## Project Context
This repository contains experiments for Hinglish language modeling, including GPT-2 and RNN baselines. The primary GPT-2 workflow is in `gpt2.ipynb` and is used for Hinglish next-word prediction via causal language modeling.

## Primary Objective
Fine-tune GPT-2 on Hinglish conversational data so the model learns code-mixed continuation patterns and can predict/generate plausible next words/tokens.

## Where Fine-Tuning Happens
- Main notebook: `gpt2.ipynb`
- Core training cells include:
  - base model/tokenizer load
  - preprocessing/tokenization
  - training args + causal LM collator
  - chunked training loop
  - checkpoint and final model saves

## Training Method (Implemented)
1. Load pretrained GPT-2 tokenizer/model from Hugging Face (`gpt2`).
2. Set tokenizer pad token to EOS token (GPT-2 has no native pad token).
3. Preprocess text by normalizing whitespace.
4. Tokenize with:
   - truncation enabled
   - fixed padding
   - `max_length=128`
5. Use causal LM data collator (`mlm=False`).
6. Read `final_dataset.csv` in chunks (`chunksize=10000`) to avoid loading all rows at once.
7. For each chunk:
   - build Hugging Face `Dataset`
   - tokenize
   - train with `Trainer`
   - evaluate on unseen validation set
   - save per-chunk model/tokenizer under `temp/`
8. Save final model/tokenizer as `gpt2_hinglish_model_final`.

## Key Data Files
- Training data: `final_dataset.csv`
- Unseen validation: `unseen_validation_set.csv`
- GPT-2 perplexity output: `transformer_perplexity_results.json`
- RNN perplexity baseline file: `rnn_perplexity_results.json`

## Model Artifact Directories
- Fine-tuning output dir: `gpt2_hinglish_finetuned`
- Checkpoints observed:
  - `gpt2_hinglish_finetuned/checkpoint-2500`
  - `gpt2_hinglish_finetuned/checkpoint-3125`
- Final exported model:
  - `gpt2_hinglish_model_final`

## Important Run Notes
- Notebook snippet shows `num_train_epochs=3`, but checkpoint metadata indicates a run completed at 5 epochs (`global_step=3125`).
- Assume final artifacts are from a later/continued run, not strictly the earliest notebook snippet.

## Inference Logic (Notebook)
The notebook includes helper functions to generate text and extract the first complete next word from model continuation for Hinglish next-word prediction demos.

## Evaluation Logic
- Perplexity is computed on:
  - training-reference subset
  - unseen validation subset
- Results are written into `transformer_perplexity_results.json`.
- Combined comparison plots with RNN models are produced in `graphs/`.

## Plot Outputs
Expected combined plots include:
- `graphs/all_models_training_perplexity.png`
- `graphs/all_models_unseen_perplexity.png`

## Practical Guidance for Future Sessions
- Start from `gpt2.ipynb` when discussing GPT-2 training behavior.
- Verify checkpoint metadata (`trainer_state.json`) before claiming exact hyperparameters for a completed run.
- Use unseen perplexity trends as the primary signal for generalization.
- Keep preprocessing/tokenization consistent across train and eval paths.

## Scope Reminder
This project uses full GPT-2 fine-tuning with causal LM objective. It does not currently use LoRA/PEFT/adapters or RLHF-style tuning.
