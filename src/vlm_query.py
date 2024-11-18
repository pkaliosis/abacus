import os
import numpy as np
import pandas as pd
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import json
import logging
from tqdm import tqdm
from PIL import Image
import argparse

sys.path.append("../")
from utils.utils import set_logger
from utils.evaluation import mae, rmse
from utils.dataset import ZSOCDataset

#os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VLMQueryExecutor:
    def __init__(self, config_path):
        # Load configurations from a JSON file
        self.config = self.load_config(config_path)

        n_gpu = self.config.get('n_gpu', 1)
        self.device = f"cuda:{str(self.config['cuda_nr'])}" if n_gpu > 0 else "cpu"

        set_logger("../snapshots/logs/", "train.log", self.config["logging"]["print_on_screen"])

    def load_config(self, config_path):
        """ Load all arguments from a JSON configuration file """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
        
        
    def main(self):
        
        df = pd.read_csv(self.config["df_path"])
        
        object_imgs_path = self.config["bboxes_path"]
        
        #test_df = df[df["split"] == "test"]
        test_df = df
        test_df["predicted_counts"] = None

        logging.info("Initializing model...")
        model = LlavaForConditionalGeneration.from_pretrained(self.config["pretrained_vlm_hf_id"]).half().to(self.device)
        processor = AutoProcessor.from_pretrained(self.config["pretrained_vlm_hf_id"], use_fast=False)
        
        logging.info("Start iterating...")
        for idx, row in tqdm(test_df.iterrows()):
            
            dataset = ZSOCDataset(
                image_folder = object_imgs_path + row["filename"][:-4],
                obj_id = row["filename"][:-4],
                obj_class = row["class"],
                obj_prompt_notation = row["prompt_notation"],
                #obj_description = row["description"],
                prompt_template = self.config["prompt_template"]
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.config["dataloader_params"]["batch_size"],
                num_workers=self.config["dataloader_params"]["num_workers"]
            )

            counter = 0
            for batch in tqdm(dataloader):
                
                img_paths = batch["img_paths"]
                prompts = batch["prompts"]
                
                images = [Image.open(path) for path in img_paths]
                
                inputs = processor(text=prompts, images=images, return_tensors="pt").to(self.device)
                
                # Generate
                generate_ids = model.generate(**inputs, max_new_tokens=self.config["generation_params"]["max_length"])
                texts = processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                answers = [text.split("ASSISTANT:")[1] for text in texts]
                for answer in answers:
                    counter += ("yes" in answer.lower())
            
            test_df.loc[idx, "predicted_counts"] = counter

            if (idx%3 == 0):
                r_mae, r_rmse = mae(test_df.head(idx)["n_objects"], test_df.head(idx)["predicted_counts"]), rmse(test_df.head(idx)["n_objects"], test_df.head(idx)["predicted_counts"])
                if not os.path.exists(os.path.dirname(self.config["output_path"])):
                    os.makedirs(os.path.dirname(self.config["output_path"]))  # Create any missing directories
                    logging.info(f"Created directories: {os.path.dirname(self.config['output_path'])}")
                test_df.to_csv(self.config["output_path"])
                logging.info(f"MAE@{idx}: {r_mae}")
                logging.info(f"RMSE@{idx}: {r_rmse}")
                logging.info(f"Saved ongoing results at {self.config['output_path']}")
        
        test_df.drop("optimized_prompts", axis=1, inplace=True)
        test_df.to_csv(self.config["output_path"])
        
        # Final evaluation
        mae_ = mae(test_df["n_objects"], test_df["predicted_counts"])
        rmse_ = rmse(test_df["n_objects"], test_df["predicted_counts"])
        
        print("Test MAE:", mae_)
        print("Test RMSE:", rmse_)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to run InstructBLIP with JSON config.")
    parser.add_argument('--config', type=str, required=False, default="../configs/vlm_config.json", help='Path to the JSON configuration file.')
    args = parser.parse_args()

    vlm_query_exec = VLMQueryExecutor(args.config)
    vlm_query_exec.main()