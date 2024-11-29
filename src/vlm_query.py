import os
import argparse
import torch
import sys
import json
import logging
import numpy as np
import pandas as pd
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForVision2Seq
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from huggingface_hub import hf_hub_download

sys.path.append("../")
from utils.utils import set_logger
from utils.evaluation import mae, rmse
from utils.dataset import ZSOCDataset
from utils.flamingo_dataset import FlamingoDataset
from utils.idefics_dataset import IdeficsDataset
from utils.qwen_dataset import QwenDataset

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
    
    def custom_collate_fn(self, batch):
        """
        Collate function to handle padding of variable-sized image inputs.
        
        Args:
            batch (list): A list of dictionaries where each dictionary contains "inputs".
        
        Returns:
            dict: A dictionary with padded tensors for "inputs".
        """
        # Extract inputs from the batch
        inputs_list = [item["inputs"] for item in batch]

        # Extract individual components of the inputs
        input_texts = [inputs["input_ids"] for inputs in inputs_list]
        images = [inputs["pixel_values"] for inputs in inputs_list]

        # Find the maximum dimensions for images
        max_height = max(image.size(-2) for image in images)
        max_width = max(image.size(-1) for image in images)

        # Pad images to the maximum dimensions
        padded_images = []
        for image in images:
            _, _, height, width = image.size()
            padding = (0, max_width - width, 0, max_height - height)  # (left, right, top, bottom)
            padded_images.append(F.pad(image, padding))

        # Stack padded images
        batch_images = torch.stack(padded_images)

        # Stack texts (already tensors, so just concatenate)
        batch_texts = torch.cat(input_texts, dim=0)

        # Return the collated batch
        return {
            "input_ids": batch_texts,
            "pixel_values": batch_images
        }

        
    def main(self):
        
        df = pd.read_csv(self.config["df_path"])
        
        object_imgs_path = self.config["bboxes_path"]
        
        #test_df = df[df["split"] == "test"]
        test_df = df
        test_df["predicted_counts"] = None

        logging.info("Initializing model...")
        if self.config["model_family"] == "llava":
            model = LlavaForConditionalGeneration.from_pretrained(self.config["pretrained_vlm_hf_id"]).half().to(self.device)
            processor = AutoProcessor.from_pretrained(self.config["pretrained_vlm_hf_id"], use_fast=False)
        elif self.config["model_family"] == "flamingo":
            model, processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
                tokenizer_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
                cross_attn_every_n_layers=2
            )

            logging.info("Loading OpenFlamingo checkpoint...")
            checkpoint_path = hf_hub_download(self.config["pretrained_vlm_hf_id"], "checkpoint.pt")
            model.load_state_dict(torch.load(checkpoint_path), strict=True)
            
            """logging.info("Reducing model precision to half...")
            model.vision_encoder.half()
            model.perceiver.half()
            model.lang_encoder.half()"""

            logging.info(f"Loading model to {self.device}...")
            model.to(self.device)
        elif self.config["model_family"] == "idefics":
            processor = AutoProcessor.from_pretrained(self.config["pretrained_vlm_hf_id"], do_image_splitting=False)
            model = AutoModelForVision2Seq.from_pretrained(
                self.config["pretrained_vlm_hf_id"],
                torch_dtype=torch.float16,
            ).to(self.device)
            model.eval()
        elif self.config["model_family"] == "qwen":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
            ).to("cuda:4")
            model.eval()
            #min_pixels = 224 * 28 * 28
            #max_pixels = 768 * 28 * 28
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")#, min_pixels=min_pixels, max_pixels=max_pixels)


        
        logging.info("Start iterating...")
        for idx, row in tqdm(test_df.iterrows()):
            if self.config["model_family"] == "llava":
                dataset = ZSOCDataset(
                    image_folder = object_imgs_path + row["filename"][:-4],
                    obj_id = row["filename"][:-4],
                    obj_class = row["class"],
                    obj_prompt_notation = row["prompt_notation"],
                    obj_description = row["generated_description"],
                    prompt_template = self.config["prompt_template"],
                    #prototypes_folder = self.config["prototypes_origin_folder"]
                    #prototypes_folder = object_imgs_path  + row["filename"][:-4] + "/prototypes"
                )
            elif self.config["model_family"] == "flamingo":
                dataset = FlamingoDataset(
                    image_folder = object_imgs_path + row["filename"][:-4],
                    obj_id = row["filename"][:-4],
                    obj_class = row["class"],
                    obj_prompt_notation = row["prompt_notation"],
                    obj_description = row["generated_description"],
                    prototypes_folder = object_imgs_path  + row["filename"][:-4] + "/prototypes",
                    image_processor = processor,
                    tokenizer = tokenizer,
                    scaling_factor=4
                )
            elif self.config["model_family"] == "idefics":
                dataset = IdeficsDataset(
                    image_folder = object_imgs_path + row["filename"][:-4],
                    obj_id = row["filename"][:-4],
                    obj_class = row["class"],
                    obj_prompt_notation = row["prompt_notation"],
                    obj_description = row["generated_description"],
                    prototypes_folder = self.config["prototypes_origin_folder"],
                    image_processor = processor,
                    scaling_factor=4
                )
            elif self.config["model_family"] == "qwen":
                dataset = QwenDataset(
                    image_folder = object_imgs_path + row["filename"][:-4],
                    obj_id = row["filename"][:-4],
                    obj_class = row["class"],
                    obj_prompt_notation = row["prompt_notation"],
                    obj_description = row["generated_description"],
                    prototypes_folder = self.config["prototypes_origin_folder"],
                    image_processor = processor,
                    scaling_factor=4
                )

            dataloader = DataLoader(
                dataset,
                batch_size=self.config["dataloader_params"]["batch_size"],
                num_workers=self.config["dataloader_params"]["num_workers"],
                #collate_fn=self.custom_collate_fn
            )

            counter = 0
            for batch in tqdm(dataloader):
                
                if self.config["model_family"] == "llava":
                    images = [Image.open(path) for path in batch["img_paths"]]
                    #prototypes = [Image.open(path) for path in batch["prototype_path"]]
                    #inp_images = [item for pair in zip([prototypes[0]] * len(images), images) for item in pair]
                    scale_factor = 4
                    upsampled_images = [image.resize(
                                        (int(image.width * scale_factor), int(image.height * scale_factor)), Image.BICUBIC
                                    ) for image in images]
                    inputs = processor(text=batch["prompts"], images=upsampled_images, return_tensors="pt").to(self.device)
                    
                    # Generate
                    generate_ids = model.generate(**inputs, max_new_tokens=self.config["generation_params"]["max_length"])
                    texts = processor.batch_decode(
                        generate_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    answers = [text.split("ASSISTANT:")[1] for text in texts]
                    print(answers)
                    for answer in answers:
                        counter += ("yes" in answer.lower())
                elif self.config["model_family"] == "flamingo":
                    generated_text = model.generate(
                        vision_x=batch["vision_input"].squeeze(1).to(self.device),
                        lang_x=batch["lang_input"]["input_ids"].squeeze(1).to(self.device),
                        attention_mask=batch["lang_input"]["attention_mask"].squeeze(1).to(self.device),
                        max_new_tokens=5,
                        num_beams=5,
                    )
                    generated_text = tokenizer.decode(generated_text[0])#.split('<image>')[-1].replace(prefix, '').replace('<|endofchunk|>', '').replace('\n', ' ')

                    print("FLAMINGO generated text:", generated_text)
                elif self.config["model_family"] == "idefics":
                    inputs = {k: v.squeeze(1).to(self.device) for k, v in batch["inputs"].items()}
                    # Generate
                    #print(batch["inputs"]["pixel_values"].shape)
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=100)
                    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    answers = [text.split("Assistant:")[4] for text in generated_texts]
                    for answer in answers:
                        counter += ("yes" in answer.lower())
                elif self.config["model_family"] == "qwen":
                    inputs = {k: v.squeeze(1).to("cuda:4") for k, v in batch["inputs"].items()}
                    # Inference
                    generated_ids = model.generate(**inputs, max_new_tokens=10)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                    ]
                    answers = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    for answer in answers:
                        counter += ("yes" in answer.lower())
            
            test_df.loc[idx, "predicted_counts"] = counter
            #print("counter:", counter)
            if (idx%5 == 0):
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