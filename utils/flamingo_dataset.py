import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Step 1: Define a custom Dataset class
class FlamingoDataset(Dataset):
    def __init__(self,
                 image_folder,
                 obj_id,
                 obj_class,
                 obj_prompt_notation,
                 obj_description,
                 prototypes_folder,
                 image_processor,
                 tokenizer,
                 scaling_factor
        ):
        self.image_folder = str(image_folder)
        self.obj_id = obj_id
        self.obj_class = obj_class
        self.obj_prompt_notation = obj_prompt_notation
        self.obj_description = obj_description
        self.prototypes_folder = prototypes_folder
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.scaling_factor = scaling_factor
        
        self.prompt = f"You are {obj_prompt_notation} specialist. I want to decide whether different images show {obj_prompt_notation}. Answer with only one word (yes or no). <image>Does this image show {obj_prompt_notation}? Yes. <|endofchunk|><image>Does this image show {obj_prompt_notation}?"
        print("PROMPT:", self.prompt)

        self.images = [path for path in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, path))]
        self.prototypes = [path for path in os.listdir(self.prototypes_folder)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_paths = self.image_folder + "/" + self.images[idx]
        prototype_path = self.prototypes_folder + "/" + self.prototypes[1]

        query_image = Image.open(img_paths)
        demo_image = Image.open(prototype_path)

        query_image_rsd = query_image.resize((int(query_image.width * self.scaling_factor), int(query_image.height * self.scaling_factor)), Image.BICUBIC)
        demo_image_rsd = demo_image.resize((int(demo_image.width * self.scaling_factor), int(demo_image.height * self.scaling_factor)), Image.BICUBIC)

        vision_x = [self.image_processor(demo_image_rsd).unsqueeze(0), self.image_processor(query_image_rsd).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        lang_x = self.tokenizer(
            [self.prompt],
            return_tensors="pt",
        )

        return {"vision_input": vision_x, "lang_input": lang_x, "prompts": self.prompt}
        