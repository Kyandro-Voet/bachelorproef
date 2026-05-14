"""
QLoRA fine-tuning voor factuur-extractie.

Gebruik:
    python train_qlora.py

Vereisten:
    pip install transformers peft bitsandbytes trl datasets accelerate

Data:
    data/train.jsonl  — rijen: {"input": "<ocr tekst>", "output": "<json>"}

Output:
    output/qlora-factuur/  — adapter-gewichten (~10-50 MB)
"""

import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from prepare_dataset import maak_dataset, splits_dataset


# ──────────────────────────────────────────────
# CONFIGURATIE
# ──────────────────────────────────────────────
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = Path("output/qlora-factuur")
DATA_PAD = Path("data/train.jsonl")

# LoRA hyperparameters
LORA_R = 16           # rank — hogere waarde = meer trainbare parameters
LORA_ALPHA = 32       # schaalcoëfficiënt (lora_alpha / r = effectieve schaal)
LORA_DROPOUT = 0.05

# Training hyperparameters
# Geoptimaliseerd voor 8GB VRAM (bv. RTX 3070):
#   effectieve batch = BATCH_SIZE * GRADIENT_ACCUMULATION = 1 * 8 = 8
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024        # utility bill prompts passen ruim in 1024 tokens


def laad_model_en_tokenizer():
    """Laad het base model in 4-bit kwantisatie en de tokenizer."""
    print(f"📥 Model laden: {BASE_MODEL}")

    # 4-bit kwantisatie configuratie (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 — beste kwaliteit
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # extra kwantisatie van schaalwaarden
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",                  # verspreid over GPU(s) automatisch
        torch_dtype=torch.bfloat16,
        use_cache=False,                    # vereist voor gradient checkpointing
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token   # Llama heeft geen pad token
    tokenizer.padding_side = "right"            # rechts padden voor causaal LM

    print(f"✅ Model geladen")
    return model, tokenizer


def configureer_lora(model):
    """Voeg LoRA adapter-lagen toe aan het model."""
    # Bereid het model voor op k-bit training (bevriest base gewichten)
    # gradient_checkpointing=True spaart ~30% VRAM ten koste van ~20% snelheid
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        # Attention projectie-lagen — hier leert het model het meest
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # toont % trainbare vs bevroren parameters
    return model


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data laden
    dataset = maak_dataset(DATA_PAD)
    train_ds, val_ds = splits_dataset(dataset)

    # Model en tokenizer laden
    model, tokenizer = laad_model_en_tokenizer()
    model = configureer_lora(model)

    # Training configuratie
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",      # kolom naam in de dataset
        report_to="none",               # zet op "wandb" voor Weights & Biases
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
    )

    # Training starten
    print("\n🚀 Training gestart...")
    trainer.train()

    # Enkel de adapter-gewichten opslaan (niet het volledige model)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\n✅ Adapters opgeslagen in: {OUTPUT_DIR}")
    print("   (Enkel de LoRA gewichten, ~10-50 MB)")


if __name__ == "__main__":
    main()
