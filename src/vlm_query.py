import numpy as np
import pandas as pd

import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import cv2
from tqdm import tqdm

from PIL import Image
import requests

sys.path.append("../")
from utils.evaluation import mae, rmse
from utils.dataset import ZSOCDataset

#os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VLMQueryExecutor:
    def __init__(self, vlm_path, df_path):
        self.vlm_path = vlm_path
        self.df_path = df_path
        
        
    def main(self):
        
        df = pd.read_csv(self.df_path)
        
        object_imgs_path = "../outputs/bboxes/"
        
        test_df = df[df["split"] == "test"][5:55]
        test_df["predicted_counts"] = None

    
        print(test_df.head())
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = LlavaForConditionalGeneration.from_pretrained(self.vlm_path).half().to("cuda")
        processor = AutoProcessor.from_pretrained(self.vlm_path, use_fast=False)
        
        for idx, row in tqdm(test_df.iterrows()):
            
            dataset = ZSOCDataset(
                image_folder = object_imgs_path + row["filename"][:-4],
                obj_id = row["filename"][:-4],
                obj_class = row["class"],
                obj_prompt_notation = row["prompt_notation"],
                obj_description = row["description"],
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=4,
                num_workers=4
            )
            counter = 0
            for batch in tqdm(dataloader):
                
                img_paths = batch["img_paths"]
                prompts = batch["prompts"]
                
                images = [Image.open(path) for path in img_paths]
                
                inputs = processor(text=prompts, images=images, return_tensors="pt").to("cuda")
                
                # Generate
                generate_ids = model.generate(**inputs, max_new_tokens=200)
                texts = processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                answers = [text.split("ASSISTANT:")[2] for text in texts]
                for answer in answers:
                    counter += ("yes" in answer.lower())
            
            test_df.loc[idx, "predicted_counts"] = counter
        
        test_df.drop("description", axis=1, inplace=True)
        test_df.to_csv('../outputs/dfs/test_df_pred.csv')
        
        # Evaluation
        mae = mae(test_df["n_objects"], test_df["predictd_counts"])
        rmse = rmse(test_df["n_objects"], test_df["predictd_counts"])
        
        print("Test MAE:", mae)
        print("Test RMSE:", rmse)
        

if __name__ == "__main__":
    vlm_query_exec = VLMQueryExecutor("llava-hf/llava-1.5-7b-hf", "../data/FSC147_384_V2/annotations/desc_annotations.csv")
    vlm_query_exec.main()