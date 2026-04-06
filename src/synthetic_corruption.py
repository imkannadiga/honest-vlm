import cv2
import numpy as np
from PIL import Image

def corrupt_region(image: Image.Image, bbox: list, method: str = "pixelate") -> Image.Image:
    """
    Takes a PIL Image and a bounding box [x_min, y_min, x_max, y_max].
    Corrupts the region and returns the modified PIL Image.
    """
    # Convert PIL to OpenCV format (numpy array)
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Extract the region of interest (ROI)
    roi = img_cv[y1:y2, x1:x2]
    
    if method == "blur":
        # Heavy Gaussian Blur
        roi = cv2.GaussianBlur(roi, (99, 99), 30)
    elif method == "pixelate":
        # Shrink down to 4x4 pixels, then blow back up to original size
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (4, 4), interpolation=cv2.INTER_LINEAR)
        roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
    # Put the corrupted ROI back into the image
    img_cv[y1:y2, x1:x2] = roi
    
    # Convert back to PIL
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)