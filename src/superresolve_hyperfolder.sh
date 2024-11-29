#!/bin/bash

# Path to the text file containing the list of directories
input_file="./superresolution/Real-ESRGAN/bboxes_folders_dave.txt"

python ./superresolution/Real-ESRGAN/inference_for_hyperfolder.py -n RealESRGAN_x4plus -i ../../../../data/add_disk1/panos/abacus/bboxes_dave/ --outscale 4
