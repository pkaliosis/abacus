import os
import sys
import ast
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import pipeline

sys.path.append("../")
from utils.utils import load_image, nms, big_box_suppress, save_bboxes
from utils.detection_result import DetectionResult

class ObjectDetector:

    def __init__(self, df_path):
        self.df_path = df_path


    def detect(
        self,
        image: Image.Image,
        labels: List[str],
        threshold: float = 0.1,
        detector_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        Save the areas of the bounding boxes as separate .jpg files with improved resolution.
        """
        # Set device to GPU if available
        device = "cuda:5" if torch.cuda.is_available() else "cpu"

        # Set the object detection model ID
        detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"

        # Load the zero-shot object detection pipeline
        object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

        # Ensure labels end with a period
        labels = [label if label.endswith(".") else label + "." for label in labels]

        # Perform object detection
        results = object_detector(image, candidate_labels=labels, threshold=threshold)
        
        # Return the detection results
        return results

    def grounded_detection(
        self,
        image: Union[Image.Image, str],
        labels: List[str],
        threshold: float = 0.05,
        output_dir: str = "../outputs/bboxes/",
        detector_id: Optional[str] = None
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        if isinstance(image, str):
            image = load_image(image)

        detections = self.detect(image, labels, threshold, detector_id)

        return detections


    def main(self):
        df = pd.read_csv(self.df_path)

        test_img_path = "../../../../data/add_disk1/panos/datasets/FSC147_384_V2/images/test/"
        
        test_df = df[df["split"].isin(["test", "test_coco"])]
        # Apply the function to create the new column
        #test_df['det_t'] = test_df['n_objects'].apply(decide_threshold)
        df['optimized_prompts'] = df['optimized_prompts'].apply(lambda x: str(x))

        print(test_df)

        detector_id = "IDEA-Research/grounding-dino-base"

        for _, row in tqdm(test_df.iterrows()):

            img = Image.open(test_img_path + row["filename"])

            detections = self.grounded_detection(
                image = img,
                labels = ast.literal_eval(row["optimized_prompts"]),
                threshold = row["threshold"],
                detector_id = detector_id
            )

            nms_idxs = nms(detections)
            nms_boxes = [detections[idx] for idx in nms_idxs]
            
            bbs_idxs = big_box_suppress(nms_boxes)
            bbs_boxes = [nms_boxes[i] for i in range(len(nms_boxes)) if not bbs_idxs.logical_not()[i]]

            print("bbs boxes:", bbs_boxes)
            print("bbs boxes shape:", bbs_boxes.shape)

            #save_bboxes(img, bbs_boxes, "../../../../data/add_disk1/panos/abacus/bboxes/" + row["filename"][:-4] + "/")



if __name__ == "__main__":
    detector = ObjectDetector("../data/FSC147_384_V2/annotations/abacus_v8.csv")
    detector.main()

