# import os
# from openai import OpenAI
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Retrieve the API key from the environment variable
# api_key = os.getenv('OPENAI_API_KEY')

# if not api_key:
#     raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# # Initialize the OpenAI client with the key
# client = OpenAI(api_key=api_key)
# completion = client.chat.completions.create(
#   model="ft:gpt-4o-mini-2024-07-18:personal:trier-20-5-prompt-excluded:ANNq48Mj",
#   messages= [
#     {"role": "system", "content": "This assistant is trained to recall the next sequence of coordinates exactly, based on learned patterns from training data. It should output only the precise continuation of the input sequence without deviation."}, 
#     {"role": "user", "content": "80 118 79 123 88 127 89 129 88 132 91 135 93 136 103 115 103 134 108 140 112 145 108 150 96 137 100 130 100 130 76 144 76 130 76 129 77 128 77 128"}
#   ]
# )
# print(completion.choices[0].message.content)


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
        # Training loss: .49 ft:gpt-4o-mini-2024-07-18:personal:trier-20-5-prompt-excluded:ANdk2pgd
        # Training loss: .96 ft:gpt-4o-mini-2024-07-18:personal:trier-20-5-prompt-excluded:ANNq48Mj 
        completion = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:trier-20-5-prompt-excluded:ANdk2pgd",
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