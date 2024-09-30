import numpy as np
import pandas as pd

import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
import os
import sys
import cv2
from tqdm import tqdm

from PIL import Image
import requests

sys.path.append("../")
from utils.evaluation import mae, rmse

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
        processor = AutoProcessor.from_pretrained(self.vlm_path)
        
        for idx, row in tqdm(test_df.iterrows()):
            obj_id = row["filename"][:-4]
            obj_class = row["class"]
            obj_prompt_notation = row["prompt_notation"]
            obj_description = row["description"]
            print("obj id:", obj_id)
            
            """
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"How does {obj_prompt_notation} look like?"},
                        ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"{obj_description}"},]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"Is this {obj_prompt_notation}? Please answer with a yes or a no."},
                        ],
                },
            ]

            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            """
            
            prompt = f"USER: How does {obj_prompt_notation} look like?\nASSISTANT: {obj_description}\nUSER: <image>\nIs this {obj_prompt_notation}? Please answer with a yes or a no.\nASSISTANT:"
            #print(prompt)
            obj_patches_path = object_imgs_path + obj_id
            counter = 0
            for file in tqdm(os.listdir(obj_patches_path)):
                
                img = Image.open(obj_patches_path + "/" + file)
                
                inputs = processor(text=prompt, images=img, return_tensors="pt").to("cuda")

                # Generate
                generate_ids = model.generate(**inputs, max_new_tokens=200)
                text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                answer = text.split("ASSISTANT:")[2]
                counter += (("Yes" in answer) or ("yes" in answer))
            
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