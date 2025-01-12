import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer


# Define base directory
BASE_DIR = os.path.join('transformer', 'llama2')

# Define subdirectories
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CODE_DIR = os.path.join(BASE_DIR, 'src')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [MODEL_DIR, CODE_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_model_and_tokenizer(model_name, device_map):
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        cache_dir=MODEL_DIR,
        token=True
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    return model, tokenizer


def create_peft_config(lora_alpha, lora_dropout, lora_r):
    return LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

def create_training_arguments(output_dir, num_train_epochs, per_device_train_batch_size,
                              gradient_accumulation_steps, learning_rate, weight_decay,
                              max_grad_norm, warmup_ratio):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        save_steps=0,
        logging_steps=25,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=True,
        max_grad_norm=max_grad_norm,
        max_steps=-1,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
    )


def train_model(model, tokenizer, dataset, peft_config, training_arguments):
    def formatting_func(example):
        return f"Input: {example['input']}\nOutput: {example['output']}"

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer, 
        args=training_arguments,
        formatting_func=formatting_func,
    )

    trainer.train()
    return trainer


def save_model(trainer, new_model):
    save_path = os.path.join(MODEL_DIR, new_model)
    trainer.model.save_pretrained(save_path)

def generate_coordinates(model, tokenizer, input_coords):
    input_tokens = tokenizer.encode(input_coords)
    input_tensor = torch.tensor(input_tokens).unsqueeze(0)
    output = model.generate(input_tensor, max_length=200)
    return tokenizer.decode(output[0].tolist())

def main():
    # Configuration
    model_name = "meta-llama/Llama-2-7b-hf"
    new_model = "Llama-2-7b-coordinate-predictor"
    output_dir = RESULTS_DIR

    # LoRA configuration
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1

    # Training configuration
    num_train_epochs = 1
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 1
    learning_rate = 2e-4
    weight_decay = 0.001
    max_grad_norm = 0.3
    warmup_ratio = 0.03
    max_seq_length=512

    # Load your coordinate dataset
    coordinate_data = [
        {"input": [80, 123, 81, 122, 81, 124, 89, 100, 93, 120], "output": [85, 110, 87, 115, 90, 118, 92, 119, 94, 121]},
        {"input": [70, 113, 71, 112, 72, 114, 79, 90, 83, 110], "output": [75, 100, 77, 105, 80, 108, 82, 109, 84, 111]},
        {"input": [60, 103, 61, 102, 62, 104, 69, 80, 73, 100], "output": [65, 90, 67, 95, 70, 98, 72, 99, 74, 101]},
        {"input": [50, 93, 51, 92, 52, 94, 59, 70, 63, 90], "output": [55, 80, 57, 85, 60, 88, 62, 89, 64, 91]},
        {"input": [40, 83, 41, 82, 42, 84, 49, 60, 53, 80], "output": [45, 70, 47, 75, 50, 78, 52, 79, 54, 81]},
    ]
    dataset = Dataset.from_list(coordinate_data)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, {"": 0})
    model.resize_token_embeddings(256)

    # Create PEFT config
    peft_config = create_peft_config(lora_alpha, lora_dropout, lora_r)

    # Create training arguments
    training_arguments = create_training_arguments(
        output_dir, num_train_epochs, per_device_train_batch_size,
        gradient_accumulation_steps, learning_rate, weight_decay,
        max_grad_norm, warmup_ratio
    )

    # Train model
    trainer = train_model(model, tokenizer, dataset, peft_config, training_arguments)

    # Save model
    save_model(trainer, new_model)

    # Generate coordinates example
    input_coords = [80, 123, 81, 122, 81, 124, 89, 100, 93, 120]
    generated_coords = generate_coordinates(model, tokenizer, input_coords)
    print(generated_coords)

if __name__ == "__main__":
    main()
