import os

def write_filenames_to_txt(folder_path, output_txt_path):
    """
    Writes the filenames of all files in the given folder to a text file.

    Parameters:
    - folder_path (str): Path to the folder containing the files.
    - output_txt_path (str): Path to the output text file.
    """
    try:
        # Get the list of all filenames in the folder
        filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Write filenames to the text file
        with open(output_txt_path, "w") as txt_file:
            for filename in filenames:
                txt_file.write(filename + "\n")
        
        print(f"Successfully wrote {len(filenames)} filenames to {output_txt_path}")
    
    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' does not exist.")
    except PermissionError:
        print(f"Error: Permission denied for accessing '{folder_path}' or writing to '{output_txt_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
if __name__ == "__main__":
    folder_path = "../../../../../data/add_disk1/panos/datasets/FSC147_384_V2/images/test/"  # Replace with the folder path
    output_txt_path = "image_filenames.txt"  # Replace with your desired output file path

    write_filenames_to_txt(folder_path, output_txt_path)