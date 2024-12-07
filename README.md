# ABACUS: Leveraging Vision-Language Models for Open-World Object Counting

This repo contains the code for my CSE527 Computer Vision Project. Please follow the instructions below if you'd like to experiment with ABACUS or reproduce its reported results.

1. Download the FSC-147 Dataset.

  You can follow the instruction found [here](https://github.com/cvlab-stonybrook/LearningToCountEverything).

2. Make sure to split the dataset into the pre-defined train, validation and test sets. You can use the script found under ```/FSC147_384_V2/preprocess/extract_counts.py``` in order to convert it to the required format.

3. Set the DAVE and ESRGAN subdirectories. More information can be found in their respective individual repos ([DAVE](https://github.com/jerpelhan/DAVE), [ESRGAN](https://github.com/xinntao/ESRGAN)).

4. Run the object detection pipeline:
   ```
   cd src/
   python detect.py
   ```

5. Superresolve the detected bounding boxes (make sure to set the path to the folder containing your bounding boxes in ```src/superresolve.sh```).
  ```
  cd src/
  ./superresolve.sh
  ```

6. Run the VLM of your choice (between IDEFICS, LLaVA, OpenFlamingo and Qwen-VL-2.5 (set up your choices in ```configs/vlm_config.json```
   ```
   cd src/
   python vlm_query.py
   ```
