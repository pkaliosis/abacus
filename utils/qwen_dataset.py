import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from qwen_vl_utils import process_vision_info

"""{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": self.demo_img_rsd,
                        },
                        {"type": "text", "text": f"Is this an image of {self.obj_prompt_notation}?"},
                    ],
                },
                {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Yes."},
                ]
                },  """

# Step 1: Define a custom Dataset class
class QwenDataset(Dataset):
    def __init__(self,
                 image_folder,
                 obj_id,
                 obj_class,
                 obj_prompt_notation,
                 obj_description,
                 prototypes_folder,
                 image_processor,
                 scaling_factor,
        ):
        self.image_folder = str(image_folder)
        self.obj_id = obj_id
        self.obj_class = obj_class
        self.obj_prompt_notation = obj_prompt_notation
        self.obj_description = obj_description
        self.prototypes_origin_folder = prototypes_folder
        self.image_processor = image_processor
        self.scaling_factor = scaling_factor

        self.images = [path for path in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, path))]
        self.prototypes_folder = os.path.join(os.path.join(self.prototypes_origin_folder, self.image_folder.split('/')[-1]), "prototypes")
        self.prototypes = [path for path in os.listdir(self.prototypes_folder)]

        self.prototype_path = self.prototypes_folder + "/" + self.prototypes[1]
        #self.prototype_path_2 = self.prototypes_folder + "/" + self.prototypes[2]
        #self.prototype_path_3 = self.prototypes_folder + "/" + self.prototypes[0]

        self.demo_image = Image.open(self.prototype_path)
        #self.demo_image_2 = Image.open(self.prototype_path_2)
        #self.demo_image_3 = Image.open(self.prototype_path_3)

        self.demo_img_rsd = self.rsz(self.demo_image, 384)
        #self.demo_img_rsd_2 = self.rsz(self.demo_image_2, 384)
        #self.demo_img_rsd_3 = self.rsz(self.demo_image_3, 384)

    def __len__(self):
        return len(self.images)

    def rsz(self, img, target_size):
        w, h = img.size
        scale = target_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))  # Scale proportionally
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            
        # Step 3: Pad to make the image square
        delta_w = target_size - new_size[0]
        delta_h = target_size - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        img_padded = ImageOps.expand(img_resized, padding, fill=(0, 0, 0))  # Use black padding
        return img_padded

    def __getitem__(self, idx):
        img_paths = self.image_folder + "/" + self.images[idx]

        query_image = Image.open(img_paths)
        query_img_rsd = self.rsz(query_image, 384)


        self.messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": query_img_rsd,
                        },
                        {"type": "text", "text": f"Is this is an image of {self.obj_prompt_notation}? Anwer with a yes or a no."},
                    ]
                }  
            ]
        
        text = self.image_processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(self.messages)
        inputs = self.image_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )


        return {"inputs": inputs}
        