from datasets import load_dataset

# Load dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[0:500]", trust_remote_code=True)

# Preprocessing steps
def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]

    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

dataset_finetune = dataset.map(formatting_prompts_func, batched=True)
