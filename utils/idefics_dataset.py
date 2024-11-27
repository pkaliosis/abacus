import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Step 1: Define a custom Dataset class
class IdeficsDataset(Dataset):
    def __init__(self,
                 image_folder,
                 obj_id,
                 obj_class,
                 obj_prompt_notation,
                 obj_description,
                 prototypes_folder,
                 image_processor,
                 scaling_factor
        ):
        self.image_folder = str(image_folder)
        self.obj_id = obj_id
        self.obj_class = obj_class
        self.obj_prompt_notation = obj_prompt_notation
        self.obj_description = obj_description
        self.prototypes_folder = prototypes_folder
        self.image_processor = image_processor
        self.scaling_factor = scaling_factor
        
        """self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Is this is an image of {obj_prompt_notation}? Anwer with a yes or a no."},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Yes."},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Is this is an image of {obj_prompt_notation}? Anwer with a yes or a no."},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Yes."},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Is this is an image of {obj_prompt_notation}? Anwer with a yes or a no."},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Yes."},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"And how about this image? Is this is an image of {obj_prompt_notation}? Anwer with a yes or a no."},
                ]
            },       
        ]"""

        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Is this is an image of {obj_prompt_notation}? Anwer with a yes or a no."},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Yes."},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"And how about this image? Is this is an image of {obj_prompt_notation}? Anwer with a yes or a no."},
                ]
            },       
        ]

        self.images = [path for path in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, path))]
        self.prototypes = [path for path in os.listdir(self.prototypes_folder)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_paths = self.image_folder + "/" + self.images[idx]
        prototype_path = self.prototypes_folder + "/" + self.prototypes[1]
        #prototype_path_2 = self.prototypes_folder + "/" + self.prototypes[2]
        #prototype_path_3 = self.prototypes_folder + "/" + self.prototypes[0]

        query_image = Image.open(img_paths)
        demo_image = Image.open(prototype_path)
        #demo_image_2 = Image.open(prototype_path_2)
        #demo_image_3 = Image.open(prototype_path_3)

        #query_image_rsd = query_image.resize((min(int(query_image.width * self.scaling_factor), 380), min(int(query_image.height * self.scaling_factor), 380)), Image.BICUBIC)
        #demo_image_rsd = demo_image.resize((min(int(demo_image.width * self.scaling_factor), 380), min(int(demo_image.height * self.scaling_factor), 380)), Image.BICUBIC)

        query_image_rsd = query_image.resize((224, 224))
        demo_image_rsd = demo_image.resize((224, 224))
        """demo_image_rsd_2 = demo_image_2.resize((224, 224), Image.BICUBIC)
        demo_image_rsd_3 = demo_image_3.resize((224, 224), Image.BICUBIC)"""

        """print("query img size:", query_image_rsd.size)
        print("demo image size:", demo_image_rsd.size)"""
        self.prompt = self.image_processor.apply_chat_template(self.messages, add_generation_prompt=True)
        #inputs = self.image_processor(text=self.prompt, images=[demo_image, demo_image_2, demo_image_3, query_image], return_tensors="pt")
        inputs = self.image_processor(text=self.prompt, images=[demo_image_rsd, query_image_rsd], return_tensors="pt")


        return {"inputs": inputs}
        