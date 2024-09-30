import csv
import json

# Function to load class information from a text file
def load_class_from_txt(class_txt_file):
    class_data = {}
    with open(class_txt_file, 'r') as f:
        for line in f:
            # Split the line by tabs
            file_name, class_name = line.strip().split('\t')
            class_data[file_name] = class_name
    return class_data

# Function to load split information from the JSON file
def load_split_from_json(split_json_file):
    split_data = {}
    with open(split_json_file, 'r') as f:
        data = json.load(f)
        for split, file_list in data.items():
            for file_name in file_list:
                split_data[file_name] = split
    return split_data

# Function to extend the CSV file with class and split information
def extend_csv_with_class_and_split(csv_file, class_txt_file, split_json_file, output_csv_file):
    # Load class information from the text file
    class_data = load_class_from_txt(class_txt_file)
    
    # Load split information from the JSON file
    split_data = load_split_from_json(split_json_file)
    
    # Read the existing CSV file and create a new extended CSV
    with open(csv_file, 'r') as infile, open(output_csv_file, 'w', newline='') as outfile:
        csvreader = csv.reader(infile)
        csvwriter = csv.writer(outfile)
        
        # Read the header and add the new "class" and "split" columns
        header = next(csvreader)
        header.extend(["class", "split"])
        csvwriter.writerow(header)
        
        # Iterate through the rows and append the class and split information
        for row in csvreader:
            file_name = row[0]  # File name is in the first column
            
            # Get the class for the file, default to "Unknown" if not found
            #image_class = class_data.get(file_name, "Unknown")
            
            # Get the split for the file, default to "Unknown" if not found
            image_split = split_data.get(file_name, "Unknown")
            
            # Append the class and split to the row and write it to the new CSV
            row.extend([image_split])
            csvwriter.writerow(row)

    print(f"CSV file extended with class and split information and saved as {output_csv_file}")

# Example usage
csv_file = '/home/ubuntu/pkaliosis/zsoc/data/FSC147_384_V2/annotations/class_count_annotation.csv'  # The CSV file from the previous script
class_txt_file = '/home/ubuntu/pkaliosis/zsoc/data/FSC147_384_V2/annotations/ImageClasses_FSC147.txt'  # The text file containing class information
split_json_file = '/home/ubuntu/pkaliosis/zsoc/data/FSC147_384_V2/annotations/Train_Test_Val_FSC_147.json'  # The JSON file containing split information ("train", "val", "test")
output_csv_file = 'extended_output.csv'  # The desired output CSV file

extend_csv_with_class_and_split(csv_file, class_txt_file, split_json_file, output_csv_file)