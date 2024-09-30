from tqdm import tqdm
import cv2
import pandas as pd
import os
from pathlib import Path

def main():
    
    df = pd.read_csv("../data/FSC147_384_V2/annotations/annotations.csv")
        
    object_imgs_path = "../outputs/sr_bboxes/"
    src_imgs = "../outputs/bboxes/"
        
    test_df = df[df["split"] == "test"][20:30]
        
    for idx, row in test_df.iterrows():
        obj_id = row["filename"][:-4]
        obj_class = row["class"]
            
        obj_patches_path = object_imgs_path + obj_id
        src_imgs_path = src_imgs + obj_id
        path = Path(object_imgs_path + obj_id)
        
        # Check if the directory exists
        if not path.exists():
            # If it does not exist, create it
            path.mkdir(parents=True, exist_ok=True)
            print(f"Directory created: {path}")
        else:
            print(f"Directory already exists: {path}")
            
        counter = 0
        for file in tqdm(os.listdir(src_imgs_path)):
            print(src_imgs_path + "/" + file)
            img = cv2.imread(src_imgs_path + "/" + file)
                
            ### SUPERRESOLUTION ###
                
            # Load the pre-trained DNN super-resolution model from OpenCV
            sr = cv2.dnn_superres.DnnSuperResImpl_create()

            # Choose a model (e.g., 'edsr', 'fsrcnn') and scale (e.g., 4x for 4x resolution)
            path = "../snapshots/edsr/EDSR_x4.pb"  # Replace with the path to the pre-trained model file
            sr.readModel(path)
            sr.setModel("edsr", 4)  # EDSR, FSRCNN, etc., with 4x scale
            
            # Enable GPU (CUDA) for processing
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # Use CUDA as backend
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)    # Use CUDA for target device

            # Perform superresolution
            result = sr.upsample(img)

            # Save or display the result
            cv2.imwrite(obj_patches_path + "/" + file, result)
            print("Saved upsampled image at:",  obj_patches_path + "/" + file, result)
                

if __name__ == "__main__":
    main()