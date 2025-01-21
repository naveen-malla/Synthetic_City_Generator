import os
import torch
import json
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from typing import Dict, List
import numpy as np
from torch.utils.data import DataLoader
import evaluate

# Directory setup
BASE_DIR = os.path.join('transformer', 't5')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

# Create directories
for directory in [MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def create_peft_config():
    """Create LoRA configuration for efficient fine-tuning"""
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )

def load_datasets(train_path: str, val_path: str) -> tuple[Dataset, Dataset]:
    """Load and prepare the datasets"""
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    
    return (
        Dataset.from_list(train_data),
        Dataset.from_list(val_data)
    )

def preprocess_function(examples: Dict) -> Dict:
    """Preprocess the data for T5"""
    model_inputs = tokenizer(
        examples["prompt"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    
    labels = tokenizer(
        examples["completion"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Custom metric for coordinate prediction
    correct_coords = 0
    total_coords = 0
    
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_coords = set(pred.strip().split('\n'))
        label_coords = set(label.strip().split('\n'))
        
        correct_coords += len(pred_coords.intersection(label_coords))
        total_coords += len(label_coords)
    
    accuracy = correct_coords / total_coords if total_coords > 0 else 0
    
    return {
        "coordinate_accuracy": accuracy
    }

def create_trainer(
    model, 
    train_dataset, 
    eval_dataset, 
    tokenizer, 
    training_args
) -> Seq2SeqTrainer:
    """Create and configure the trainer"""
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

def generate_sample_output(model, tokenizer, sample):
    """Generate and print sample output"""
    inputs = tokenizer(sample["prompt"], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        temperature=0.7,
        no_repeat_ngram_size=2
    )
    
    print("\nSample Generation:")
    print("Input:", sample["prompt"])
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("Expected:", sample["completion"])

def main():
    try:
        # Model configuration
        model_name = "google/flan-t5-large"
        new_model_name = "t5-coordinate-predictor"
        
        # Load datasets
        train_path = os.path.join(DATASET_DIR, 'train_t5_coordinates.json')
        val_path = os.path.join(DATASET_DIR, 'val_t5_coordinates.json')
        train_dataset, eval_dataset = load_datasets(train_path, val_path)
        
        print(f"Loaded {len(train_dataset)} training examples and {len(eval_dataset)} validation examples")
        
        # Initialize tokenizer and model
        global tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Load model with quantization for efficient fine-tuning
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            load_in_8bit=True
        )
        
        # Prepare model for PEFT
        model = prepare_model_for_kbit_training(model)
        peft_config = create_peft_config()
        model = get_peft_model(model, peft_config)
        
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        
        # Process datasets
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=RESULTS_DIR,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,
            evaluation_strategy="steps",
            eval_steps=100,
            logging_steps=100,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
            save_steps=1000,
            bf16=True,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="coordinate_accuracy",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=512,
            generation_num_beams=4,
            include_inputs_for_metrics=True,
        )
        
        # Create and start trainer
        trainer = create_trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            training_args=training_args
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the final model
        save_path = os.path.join(MODEL_DIR, new_model_name)
        trainer.save_model(save_path)
        print(f"Model saved to {save_path}")
        
        # Generate sample output
        sample = eval_dataset[0]
        generate_sample_output(model, tokenizer, sample)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
