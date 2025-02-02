import csv
import json

# Load the coordinate predictions from a JSON file
input_file = 'predicted_results_synthetic.json'
output_file = 'predicted_coordinates_synthetic.csv'

# Open and load the JSON data (list of strings in "y x" format)
with open(input_file, 'r') as f:
    coordinates_list = json.load(f)

# Prepare the CSV output file
with open(output_file, 'w', newline='') as csvfile:
    # CSV column names
    fieldnames = ['osmnid', 'y', 'x']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the CSV header row
    writer.writeheader()

    # Initialize osmnid counter
    osmnid = 0
    
    # Process each pair of coordinates
    for coordinate in coordinates_list:
        try:
            # Split each coordinate string into y and x (latitude/longitude)
            y, x = map(float, coordinate.split())
            
            # Create a dictionary for a row of data
            row = {
                "osmnid": osmnid,
                "y": y,
                "x": x
            }

            # Write the row to the CSV
            writer.writerow(row)

            # Increment the osmnid counter
            osmnid += 1

        except (ValueError, IndexError) as e:
            print(f"Skipping due to error in handling: {e}")
            continue

print(f"'{output_file}' has been created successfully.")