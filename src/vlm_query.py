import numpy as np
import pandas as pd

import transformers
import torch
import os

from PIL import Image

from transformers import IdeficsForVisionText2Text, AutoProcessor

class VLMQueryExecutor:
    def __init__(self, vlm_path, df_path):
        self.vlm_path = vlm_path
        self.df_path = df_path
        
        
    def main(self):
        
        df = pd.read_csv(self.df_path)
        
        object_imgs_path = "../outputs/bboxes/"
        
        test_df = df[df["split"] == "test"][20:30]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = "HuggingFaceM4/idefics-9b"
        model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.int64).to(device)
        processor = AutoProcessor.from_pretrained(checkpoint)
        
        for idx, row in test_df.iterrows():
            obj_id = row["filename"][:-4]
            obj_class = row["class"]
            
            obj_patches_path = object_imgs_path + obj_id
            for file in os.listdir(obj_patches_path):
                img = Image.open(obj_patches_path + "/" + file)
                # We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.
                prompts = [
                    [
                        "\nUser:",
                        img,
                        f"Is this a single {obj_class}? Please answer with a yes or a no despite being unsure.<end_of_utterance>",

                        "\nAssistant:",
                    ],
                ]
                
                inputs = processor(prompts, return_tensors="pt").to(device)
                
                # Generation args
                bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

                generated_ids = model.generate(**inputs, bad_words_ids=bad_words_ids, max_length=100)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                for i, t in enumerate(generated_text):
                    print(f"{i}:\n{t}\n")
            

if __name__ == "__main__":
    vlm_query_exec = VLMQueryExecutor("../data/FSC147_384_V2/annotations/annotations.csv", "../data/FSC147_384_V2/annotations/annotations.csv")
    vlm_query_exec.main()