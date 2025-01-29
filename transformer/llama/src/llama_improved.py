import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig
from trl import SFTTrainer

# Define base directory and subdirectories
BASE_DIR = os.path.join('transformer', 'llama')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CODE_DIR = os.path.join(BASE_DIR, 'src')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

# Create directories if they don't exist
for directory in [MODEL_DIR, CODE_DIR, RESULTS_DIR, DATASET_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_prompt_template():
    template_path = os.path.join(CODE_DIR, 'llama_prompt.txt')
    with open(template_path, 'r') as f:
        return f.read()

def load_model_and_tokenizer(model_name, device_map):
    # Check for GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_map = "auto"
    else:
        device = torch.device('cpu')
        device_map = {"": device}

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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer, device

def create_peft_config():
    return LoraConfig(
        lora_alpha=32,         # Increase alpha
        lora_dropout=0.05,     # Lower dropout
        r=128,                 # Increase rank
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]  # Target specific modules
    )

def create_training_arguments(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=12,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        save_steps=1000,
        logging_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        learning_rate=1e-4,
        weight_decay=0.1,
        fp16=True,
        max_grad_norm=0.5,
        max_steps=-1,
        warmup_ratio=0.15,
        group_by_length=True,
        lr_scheduler_type="cosine"
    )



def create_early_stopping_callback():
    return EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    )

def calculate_statistics(dataset):
    """Calculate basic statistics of the dataset"""
    lengths = [len(str(item['prompt'])) for item in dataset]
    return {
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths)
    }

def compare_distributions(train_stats, val_stats, threshold=0.2):
    """Compare statistics between train and validation sets"""
    relative_diff = abs(train_stats['mean_length'] - val_stats['mean_length']) / train_stats['mean_length']
    if relative_diff > threshold:
        print(f"Warning: Large difference in data distributions detected: {relative_diff:.2f}")
    return relative_diff < threshold

def validate_data_distribution(train_dataset, val_dataset):
    """Ensure data distribution is similar across splits"""
    train_stats = calculate_statistics(train_dataset)
    val_stats = calculate_statistics(val_dataset)
    return compare_distributions(train_stats, val_stats)

def log_metrics(trainer, eval_results):
    """Log training metrics"""
    metrics = {
        "eval_loss": eval_results["eval_loss"],
        "eval_runtime": eval_results["eval_runtime"],
        "epoch": eval_results["epoch"]
    }
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return metrics


def train_model(model, tokenizer, train_dataset, val_dataset, peft_config, training_arguments, callbacks):
    def formatting_func(example):
        return [f"{example['prompt']}{example['completion']}"]  

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
        formatting_func=formatting_func,
        callbacks=callbacks
    )
    
    trainer.train()
    return trainer

def save_model(trainer, new_model):
    save_path = os.path.join(MODEL_DIR, new_model)
    trainer.model.save_pretrained(save_path)
    print(f"\nModel saved to: {save_path}")

def generate_coordinates(model, tokenizer, test_example, device):
    input_text = test_example['prompt']
    input_tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(
        input_tokens, 
        max_length=500,
        num_return_sequences=1,
        temperature=0.1,        # Lower temperature for more deterministic output
        top_p=0.9,             # Nucleus sampling parameter
        top_k=50,              # Limit vocabulary choices
        do_sample=False,       # Disable sampling for deterministic output
        num_beams=5,           # Use beam search
        early_stopping=True,   # Stop when valid output is generated
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def load_datasets(train_path, val_path):
    try:
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        with open(val_path, 'r') as f:
            val_data = json.load(f)
        return Dataset.from_list(train_data), Dataset.from_list(val_data)
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        raise

def create_test_example(dataset):
    test_example = dataset[0]
    print("\nTest Example:")
    print("Prompt:", test_example['prompt'])
    print("Expected Completion:", test_example['completion'])
    return test_example

def main():
    try:
        # Configuration
        model_name = "meta-llama/Llama-2-13b-hf"
        new_model = "Llama-2-13b-coordinate-predictor"
        output_dir = RESULTS_DIR


        # Load datasets
        train_path = os.path.join(DATASET_DIR, 'train_10_50_nodes_world_center.json')
        val_path = os.path.join(DATASET_DIR, 'valid_10_50_nodes_world_center.json')
        train_dataset, val_dataset = load_datasets(train_path, val_path)

        # Validate data distribution
        validate_data_distribution(train_dataset, val_dataset)

        # Create test example from validation set
        test_example = create_test_example(val_dataset)

        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer(model_name, {"": 0})

        # Create PEFT config
        peft_config = create_peft_config()

        # Create callbacks
        early_stopping = create_early_stopping_callback()

        # Create training arguments
        training_arguments = create_training_arguments(output_dir)


        # Train model
        trainer = train_model(
            model, 
            tokenizer, 
            train_dataset, 
            val_dataset, 
            peft_config, 
            training_arguments,
            callbacks=[early_stopping]
        )

        # Log evaluation metrics
        eval_results = trainer.evaluate()
        metrics = log_metrics(trainer, eval_results)

        # Save model
        save_model(trainer, new_model)

        # Generate coordinates using test example
        generated_output = generate_coordinates(model, tokenizer, test_example, device)
        print("\nGenerated Output:", generated_output)
        print("\nComparison:")
        print("Expected:", test_example['completion'])
        print("Generated:", generated_output)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
