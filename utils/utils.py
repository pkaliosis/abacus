import os
import torch
import torchvision
import requests
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Any, Optional

from .detection_result import DetectionResult

def set_logger(log_path, file_name, print_on_screen):
    """
    Write logs to checkpoint and console
    """

    log_file = os.path.join(log_path, file_name)

    logging.basicConfig(
        format="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w",
    )
    if print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image


def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


def decide_threshold(
        n_objects
    ):
    if 0 <= n_objects < 25:
        return 0.1
    elif 25 <= n_objects < 50:
        return 0.05
    elif n_objects >= 50:
        return 0.001
    
def nms(
        detections,
        threshold: float = 0.25
    ):
    scores = [d["score"] for d in detections]
    boxes = torch.tensor([list(d["box"].values()) for d in detections]).to(torch.float32)
    return torchvision.ops.nms(boxes, torch.tensor(scores).to(torch.float32), threshold)


def big_box_suppress(
        detections
    ):
    boxes = torch.tensor([list(d["box"].values()) for d in detections]).to(torch.float32)
    xmin = boxes[:,0].unsqueeze(-1)
    ymin = boxes[:,1].unsqueeze(-1)
    xmax = boxes[:,2].unsqueeze(-1)
    ymax = boxes[:,3].unsqueeze(-1)
    sz = boxes.shape[0]

    keep_ind = ((xmax >= xmax.T)&(ymax >= ymax.T)&(xmin <= xmin.T)&(ymin <= ymin.T)&(torch.eye(sz).logical_not())).any(dim=-1).logical_not()
    return keep_ind

def save_bboxes(
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

        print(f"Saved {len(results)} detected objects to {output_dir}\n")

def plot_bboxes(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    figsize: tuple = (10, 10)
) -> None:
    """
    Plots the predicted bounding boxes over the original image.

    Args:
        image (PIL.Image.Image): The original image on which to plot the bounding boxes.
        detections (List[Dict[str, Any]]): List of detection results, where each detection is
                                           a dictionary containing 'label', 'box', and 'score'.
        figsize (tuple): The size of the figure to display the image and bounding boxes.
    """
    # Convert the PIL image to a numpy array for plotting
    image_np = np.array(image)

    # Create a figure and axis to plot on
    fig, ax = plt.subplots(1, figsize=figsize)

    # Display the original image
    ax.imshow(image_np)

    # Loop through each detection result and plot the bounding boxes
    for detection in detections:
        #label = detection['label']
        #score = detection['score']
        box = detection['box']

        # Extract bounding box coordinates
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']

        # Create a rectangle patch for the bounding box
        rect = patches.Rectangle(
            (xmin, ymin),  # (x, y) - bottom-left corner
            xmax - xmin,   # width
            ymax - ymin,   # height
            linewidth=2, edgecolor='red', facecolor='none'
        )

        # Add the rectangle patch to the image
        ax.add_patch(rect)

        # Add label and score above the bounding box
        """ax.text(
            xmin, ymin - 10, f'{label} ({score:.2f})',
            color='red', fontsize=12, backgroundcolor='white'
        )"""

    # Turn off the axis
    plt.axis('off')

    # Show the plot
    plt.show()
