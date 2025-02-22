import os
from openai import OpenAI
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize the OpenAI client with the key
client = OpenAI(api_key=api_key)

# Function to make predictions for each example
def predict_coordinates(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]

    results = []
    for entry in data:
        # Extracting only the 'system' and 'user' messages from each example
        messages = [
            {"role": "system", "content": entry["messages"][0]["content"]},
            {"role": "user", "content": entry["messages"][1]["content"]}
        ]

        # API call to get the prediction
        completion = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:trier-10-5-prompt-excluded:AO6IWWno",
            messages=messages
        )
        
        # Save the output content for each prediction
        predicted_output = completion.choices[0].message.content
        results.append({
            "input": entry["messages"][1]["content"],
            "expected_output": entry["messages"][2]["content"],
            "predicted_output": predicted_output
        })
        
        # Display result to monitor progress
        print(f"Input: {entry['messages'][1]['content']}")
        print(f"Expected Output: {entry['messages'][2]['content']}")
        print(f"Predicted Output: {predicted_output}\n")

    return results

# Run predictions on all training examples
file_path = "coordinate_data.jsonl"
results = predict_coordinates(file_path)

# Optionally save results to a file
output_file = 'predictions_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Predictions completed and saved to '{output_file}'.")