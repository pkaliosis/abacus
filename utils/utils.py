from PIL import Image
from typing import List, Dict, Any, Optional

from .detection_result import DetectionResult

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


def decide_threshold(n_objects):
    if 0 <= n_objects < 25:
        return 0.1
    elif 25 <= n_objects < 50:
        return 0.05
    elif n_objects >= 50:
        return 0.001