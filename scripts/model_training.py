from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply LoRA
model_lora = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model_lora,
    tokenizer=tokenizer,
    train_dataset=dataset_finetune,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        output_dir="outputs",
    ),
)

# Start training
trainer.train()