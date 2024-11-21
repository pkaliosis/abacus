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
                 obj_description,
                 prototypes_folder
        ):
        self.image_folder = str(image_folder)
        self.obj_id = obj_id
        self.obj_class = obj_class
        self.obj_prompt_notation = obj_prompt_notation
        self.obj_description = obj_description
        #self.prompt = prompt_template.format(obj_prompt_notation=self.obj_prompt_notation[0])
        #self.prototypes_folder = prototypes_folder
        
            #self.prompt = f"USER: How does {obj_prompt_notation} look like?\n ASSISTANT: {self.obj_description}.\n USER: <image> Does this image show {obj_prompt_notation}? Please answer with a yes or a no.\nASSISTANT:"
        self.prompt = f"USER: <image> Does this image show {obj_prompt_notation}? Please answer with a yes or a no.\nASSISTANT:"
        print("PROMPT:", self.prompt)

        self.images = [path for path in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, path))]
        #self.prototypes = [path for path in os.listdir(self.prototypes_folder)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_paths = self.image_folder[0] + "/" + self.images[idx]
        #prototype_path = self.prototypes_folder + "/" + self.prototypes[1]
        return {"img_paths": img_paths, "prompts": self.prompt}#, "prototype_path": prototype_path}
        