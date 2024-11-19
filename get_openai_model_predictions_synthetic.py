import os
from openai import OpenAI
import json
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize the OpenAI client with the key
client = OpenAI(api_key=api_key)

# Function to validate coordinate pairs (ensures that only number pairs are passed)
def is_valid_coordinate(coord_pair):
    try:
        lat, lon = map(int, coord_pair.split())  # Expect two space-separated integers
        return True
    except (ValueError, TypeError):
        return False

# Function to extract valid coordinates from the predicted output
def extract_valid_coordinates(text):
    candidates = text.strip().split()
    pairs = [" ".join(candidates[i:i + 2]) for i in range(0, len(candidates), 2)]
    valid_coords = [pair for pair in pairs if is_valid_coordinate(pair)]
    return valid_coords

# Retry logic on invalid outputs
def generate_with_retries(model_name, messages, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            response = completion.choices[0].message.content
            valid_coordinates = extract_valid_coordinates(response)
            if valid_coordinates:
                return valid_coordinates
            else:
                raise ValueError("No valid coordinates in the output.")
        except Exception as e:
            retries += 1
            print(f"Retry {retries}/{max_retries} due to error: {e}")
            time.sleep(1)  # Adding small delay before retrying
    return []

# Main loop for generating new coordinates
def generate_full_sequence(model_name, initial_input, total_pairs=1400):
    current_input = initial_input[:]  # Copy of the first 10 pairs for modification
    generated_coordinates = initial_input[:]  # Store all generated coordinates
    num_coordinates = len(generated_coordinates)

    while num_coordinates < total_pairs:
        # Prepare "system" and "user" message following training format
        messages = [
            {"role": "system", "content": "This assistant is trained to recall the next sequence of coordinates exactly, based on learned patterns."},
            {"role": "user", "content": " ".join(current_input)}  # User content is space-separated coordinates
        ]

        # Call the model and process its output
        next_coordinates = generate_with_retries(model_name, messages)

        if not next_coordinates:
            print(f"Stopping: Unable to get valid predictions from model after several retries for input: {current_input}")
            break  # Exits if valid outputs couldn't be generated after retries

        # Append the valid predicted coordinates
        generated_coordinates.extend(next_coordinates)

        # Update current input for the next cycle: last 5 from previous + new 5 predicted
        current_input = current_input[-5:] + next_coordinates

        # Update number of generated coordinates
        num_coordinates = len(generated_coordinates)
        print(f"Generated {num_coordinates}/{total_pairs} coordinates so far.")

    return generated_coordinates

# Initial input (first 10 coordinate pairs)
initial_input = [
    "80 118", "79 123", "88 127", "89 129", "88 132",
    "91 135", "93 136", "103 115", "103 134", "108 140"
]

# Generate predictions until we reach 1400 coordinates
generated_coordinates = generate_full_sequence(
    model_name="ft:gpt-4o-mini-2024-07-18:personal:trier-10-5-prompt-excluded:AO6IWWno", 
    initial_input=initial_input,
    total_pairs=1400
)

# Save output to a JSON file
output_file = 'generated_coordinates_synthetic.json'
with open(output_file, 'w') as f:
    json.dump(generated_coordinates, f, indent=4)

print(f"Generated coordinates saved to '{output_file}'. Total generated: {len(generated_coordinates)}")