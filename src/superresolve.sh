#!/bin/bash

# Path to the text file containing the list of directories
input_file="./superresolution/Real-ESRGAN/bboxes_folders.txt"

# Loop through each line in the text file
while IFS= read -r line; do
    # Construct the command, replacing 3646 with the current line
    python ./superresolution/Real-ESRGAN/inference_for_hyperfoldern.py -n RealESRGAN_x4plus -i ../../../../data/add_disk1/panos/abacus/bboxes/"$line"/ --output ../../../../data/add_disk1/panos/abacus/sr_bboxes/"$line"/ --outscale 4
done < "$input_file"