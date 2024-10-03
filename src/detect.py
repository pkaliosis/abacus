import os
import sys
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import pipeline

sys.path.append("../")
from utils.utils import load_image, decide_threshold
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
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set the object detection model ID
        detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"

        # Load the zero-shot object detection pipeline
        object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

        # Ensure labels end with a period
        labels = [label if label.endswith(".") else label + "." for label in labels]
        #print("labels:", labels)
        
        # Perform object detection
        results = object_detector(image, candidate_labels=labels, threshold=threshold)

        # Return the detection results
        return results

    def save_bboxes(
        self,
        image,
        results,
        output_dir : str =  "../outputs/bboxes/unknown/"
    ):

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Convert the PIL image to a NumPy array for manipulation
        image_np = np.array(image)

        # Process the detected bounding boxes and save the cropped areas
        for i, result in enumerate(results):
            # Get bounding box coordinates
            box = result["box"]  # Expected to be a dict with "xmin", "ymin", "xmax", "ymax"

            # Extract bounding box coordinates
            xmin, ymin, xmax, ymax = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])

            # Optionally add some padding around the bounding box (optional, here it's 10 pixels padding)
            padding = 5
            xmin = max(0, xmin - padding)
            ymin = max(0, ymin - padding)
            xmax = min(image.width, xmax + padding)
            ymax = min(image.height, ymax + padding)

            # Crop the image based on the bounding box
            cropped_image = image.crop((xmin, ymin, xmax, ymax))

            # Save the upscaled cropped image as a .jpg file
            output_path = os.path.join(output_dir, f"detected_object_{i + 1}.png")
            cropped_image.save(output_path, "PNG", optimizer=True)

        print(f"Saved {len(results)} detected objects to {output_dir}")



    def grounded_segmentation(
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


    def nms(
        self,
        detections,
        threshold: float = 0.25
    ):
        scores = [d["score"] for d in detections]
        boxes = torch.tensor([list(d["box"].values()) for d in detections]).to(torch.float32)
        return torchvision.ops.nms(boxes, torch.tensor(scores).to(torch.float32), threshold)


    def big_box_suppress(
        self,
        detections
        ):
        boxes = torch.tensor([list(d["box"].values()) for d in detections]).to(torch.float32)
        xmin = boxes[:,0].squeeze(-1)
        ymin = boxes[:,1].squeeze(-1)
        xmax = boxes[:,2].squeeze(-1)
        ymax = boxes[:,3].squeeze(-1)
        sz = boxes.shape[0]

        boxes = torch.tensor([list(d["box"].values()) for d in detections]).to(torch.float32)
        xmin = boxes[:,0].unsqueeze(-1)
        ymin = boxes[:,1].unsqueeze(-1)
        xmax = boxes[:,2].unsqueeze(-1)
        ymax = boxes[:,3].unsqueeze(-1)
        sz = boxes.shape[0]

        keep_ind = ((xmax >= xmax.T)&(ymax >= ymax.T)&(xmin <= xmin.T)&(ymin <= ymin.T)&(torch.eye(sz).logical_not())).any(dim=-1).logical_not()
        return keep_ind


    def main(self):
        df = pd.read_csv(self.df_path)

        test_img_path = "../data/FSC147_384_V2/images/test/"
        
        test_df = df[df["split"] == "test_coco"]
        # Apply the function to create the new column
        test_df['det_t'] = test_df['n_objects'].apply(decide_threshold)
        
        print(test_df)

        detector_id = "IDEA-Research/grounding-dino-base"

        for idx, row in test_df.iterrows():

            img = Image.open(test_img_path + row["filename"])
            labels = [row["class"], row["description"]]
            detections = self.grounded_segmentation(
                image = img,
                labels = labels,
                threshold = row["det_t"],
                detector_id = detector_id
            )

            nms_idxs = self.nms(detections)
            nms_boxes = [detections[idx] for idx in nms_idxs]
            
            #bbs_idxs = self.big_box_suppress(nms_boxes)
            #bbs_boxes = [nms_boxes[i] for i in range(len(nms_boxes)) if not bbs_idxs.logical_not()[i]]


            self.save_bboxes(img, nms_boxes, "../outputs/bboxes/" + row["filename"][:-4] + "/")



if __name__ == "__main__":
    detector = ObjectDetector("../data/FSC147_384_V2/annotations/desc_annotations_with_coco_2.csv")
    detector.main()

