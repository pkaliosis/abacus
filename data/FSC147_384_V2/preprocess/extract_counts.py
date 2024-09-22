import json
import csv

# Function to parse the JSON file and save results to CSV
def parse_json_and_save_to_csv(json_file, csv_file):
    # Load the JSON data from the file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Open the CSV file for writing
    with open(csv_file, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csvwriter = csv.writer(csvfile)
        
        # Write the header row
        csvwriter.writerow(['filename', 'n_objects'])
        
        # Loop through each entry in the JSON file
        for file_name, file_data in data.items():
            # Get the number of points under the "points" key
            num_points = len(file_data.get('points', []))
            
            # Write the file name and number of points to the CSV
            csvwriter.writerow([file_name, num_points])

    print(f"Data has been saved to {csv_file}")

# Example usage
json_file = '../data/annotation_FSC147_384.json'  # Replace with your actual JSON file path
csv_file = '../data/FSC147_384_V2/count_annotation.csv'  # Replace with your desired CSV file path
parse_json_and_save_to_csv(json_file, csv_file)