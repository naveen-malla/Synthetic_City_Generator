import os
import torch
import json
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
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
    predictions = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Clip predictions to valid token range
    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
    
    # Handle padding and invalid tokens
    predictions = np.where(predictions < tokenizer.vocab_size, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rest of your metric computation
    correct_coords = 0
    total_coords = 0
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_coords = set(pred.strip().split('\n'))
        label_coords = set(label.strip().split('\n'))
        correct_coords += len(pred_coords.intersection(label_coords))
        total_coords += len(label_coords)
    
    accuracy = correct_coords / total_coords if total_coords > 0 else 0
    return {"coordinate_accuracy": accuracy}




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
    try:
        input_ids = torch.tensor(sample['input_ids']).unsqueeze(0)
        input_ids = input_ids.to(model.device)
        
        # Fix: Use keyword arguments for generate
        outputs = model.generate(
            input_ids=input_ids,  # Note the keyword argument
            max_length=256,
            num_beams=4,
            temperature=0.7,
            no_repeat_ngram_size=2
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        expected_text = tokenizer.decode(sample['labels'], skip_special_tokens=True)
        
        print("\nSample Generation:")
        print("Generated:", generated_text)
        print("Expected:", expected_text)
    except Exception as e:
        print(f"Error in generate_sample_output: {str(e)}")



def create_training_arguments(output_dir):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=10,
        save_steps=50,
        bf16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="coordinate_accuracy",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=512,
        generation_num_beams=4,
        remove_unused_columns=True,
        save_safetensors=False 
)


def main():
    try:
        # Model configuration
        model_name = "google/flan-t5-large"
        new_model_name = "t5-coordinate-predictor"
        
        # Load datasets
        train_path = os.path.join(DATASET_DIR, 'train_t5_coordinates.json')
        val_path = os.path.join(DATASET_DIR, 'valid_t5_coordinates.json')
        train_dataset, eval_dataset = load_datasets(train_path, val_path)
        
        # Take a small subset for testing
        train_dataset = train_dataset.select(range(500))  
        eval_dataset = eval_dataset.select(range(100))     
        
        print(f"Loaded {len(train_dataset)} training examples and {len(eval_dataset)} validation examples")
        
        # Initialize tokenizer and model
        global tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config
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
        
        # Create training arguments
        training_args = create_training_arguments(RESULTS_DIR)
        
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