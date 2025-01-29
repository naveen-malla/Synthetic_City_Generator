import os
import json
import torch
import random
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_test_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def load_trained_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_coordinates(model, tokenizer, test_example, device):
    try:
        input_text = test_example['prompt']
        input_tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        output = model.generate(
            input_tokens,
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True
        )
        
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"Generation failed: {str(e)}"

def main():
    # Define paths
    BASE_DIR = os.path.join('transformer', 'llama')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    OUTPUT_FILE = os.path.join(BASE_DIR, 'results', 'transformer_output.txt')
    
    # Load dataset
    dataset_path = os.path.join(DATASET_DIR, 'train_10_50_nodes_world_center.json')
    dataset = load_test_dataset(dataset_path)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    model_path = os.path.join(MODEL_DIR, "Llama-2-13b-coordinate-predictor")
    model, tokenizer = load_trained_model(model_path, device)
    
    # Select 5 random examples
    dataset_size = len(dataset)
    random_indices = random.sample(range(dataset_size), 5)
    test_examples = [dataset[i] for i in random_indices]
    
    # Generate and save outputs
    with open(OUTPUT_FILE, 'w') as f:
        for i, example in enumerate(test_examples, 1):
            f.write(f"\n{'='*50}\n")
            f.write(f"Test City {i}\n")
            f.write(f"{'='*50}\n\n")
            
            f.write("Input Prompt:\n")
            f.write(f"{example['prompt']}\n\n")
            
            f.write("Expected Output:\n")
            f.write(f"{example['completion']}\n\n")
            
            generated_output = generate_coordinates(model, tokenizer, example, device)
            f.write("Generated Output:\n")
            f.write(f"{generated_output}\n")

if __name__ == "__main__":
    main()
