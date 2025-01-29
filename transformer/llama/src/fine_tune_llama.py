import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
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
    
    # gradient checkpointing for memory efficiency
    #model.gradient_checkpointing_enable()
    return model, tokenizer, device

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
        save_steps=1000,
        logging_steps=100,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=True,
        max_grad_norm=max_grad_norm,
        max_steps=-1,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type="cosine",
    )

def train_model(model, tokenizer, dataset, peft_config, training_arguments):
    def formatting_func(example):
        return [f"{example['prompt']}{example['completion']}"]  

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
        formatting_func=formatting_func
    )
    
    trainer.train()
    return trainer

def save_model(trainer, tokenizer, new_model):
    save_path = os.path.join(MODEL_DIR, new_model)
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def generate_coordinates(model, tokenizer, test_example, device):
    input_text = test_example['prompt']
    input_tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(
        input_tokens, 
        max_length=500,  
        num_return_sequences=1
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def create_test_example(dataset):
    # Get the first example from the dataset
    test_example = dataset[0]
    print("\nTest Example:")
    print("Prompt:")
    print(test_example['prompt'])
    print("\nExpected Completion:")
    print(test_example['completion'])
    return test_example

def main():
   # Configuration
    model_name = "meta-llama/Llama-2-13b-hf"
    new_model = "Llama-2-13b-coordinate-predictor"
    output_dir = RESULTS_DIR

    # LoRA configuration
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1

    # Training configuration
    num_train_epochs = 8
    per_device_train_batch_size = 3
    gradient_accumulation_steps = 2
    learning_rate = 2e-4
    weight_decay = 0.1
    max_grad_norm = 1.0
    warmup_ratio = 0.1
    max_seq_length = 2048

    # Load the dataset
    dataset_path = os.path.join(DATASET_DIR, 'train_10_50_nodes_world_center.json')
    dataset = load_dataset(dataset_path)

    # Create test example before training
    test_example = create_test_example(dataset)

    # Load model and tokenizer with device
    model, tokenizer, device = load_model_and_tokenizer(model_name, {"": 0})

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

    # Generate coordinates using the test example
    generated_output = generate_coordinates(model, tokenizer, test_example, device)
    print("\nGenerated Output:")
    print(generated_output)

    # Optional: Compare with expected completion
    print("\nComparison:")
    print("Expected:", test_example['completion'])
    print("Generated:", generated_output)

if __name__ == "__main__":
    main()
