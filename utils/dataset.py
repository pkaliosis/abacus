import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Step 1: Define a custom Dataset class
class ZSOCDataset(Dataset):
    def __init__(self,
                 image_folder,
                 obj_id,
                 obj_class,
                 obj_prompt_notation,
                 obj_description
        ):
        
        self.image_folder = str(image_folder),
        self.obj_id = obj_id,
        self.obj_class = obj_class,
        self.obj_prompt_notation = obj_prompt_notation,
        self.obj_description = obj_description
        
        self.prompt = f"USER: <image>\nIs this {obj_prompt_notation}? Please answer with a yes or a no.\nASSISTANT:"

        self.images = [path for path in os.listdir(image_folder)]

    def __len__(self):
        return len(os.listdir(self.image_folder[0]))

    def __getitem__(self, idx):
        img_paths = self.image_folder[0] + "/" + self.images[idx]
        return {"img_paths": img_paths, "prompts": self.prompt}
    
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