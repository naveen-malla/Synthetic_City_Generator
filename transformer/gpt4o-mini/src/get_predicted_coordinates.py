import csv
import json

# Define the expected length of each output sequence
expected_output_length = 5  # Change this to match the actual required length for predictions

# Load predictions from JSON file
with open('predictions_results.json', 'r') as f:
    predictions = json.load(f)

# Prepare the CSV output file
with open('predicted_coordinates.csv', 'w', newline='') as csvfile:
    fieldnames = ['osmnid', 'y', 'x']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Initialize osmnid counter
    osmnid = 0

    # Process each entry in predictions
    for entry in predictions:
        # Extract input sequence
        input_sequence = entry["input"].split()
        
        # Extract predicted output, and truncate if it exceeds the expected length
        predicted_output = entry["predicted_output"].split()
        if len(predicted_output) > expected_output_length * 2:
            predicted_output = predicted_output[:expected_output_length * 2]

        # Combine input and predicted output for the complete sequence
        complete_sequence = input_sequence + predicted_output
        
        # Write rows for each pair in the combined complete sequence
        for i in range(0, len(complete_sequence), 2):

            try:
                row = {
                    "osmnid": osmnid,
                    "y": float(complete_sequence[i]),
                    "x": float(complete_sequence[i + 1]),
                }
                writer.writerow(row)
                osmnid += 1
            except (ValueError, IndexError) as e:
                print(f"Skipping due to error: {e}")
                continue

print("predicted_coordinates.csv has been created successfully.")