# Role 4 - Matteo Godinez

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from data import load_pair

def generate_report_figure(image_id, resolution="full"):
    """
    Generates a 1x4 side-by-side comparison figure:
    Color | Canny | U-Net | True Sketch
    """
    # Load ground truth inputs using Role 1's loader
    color, true_mask = load_pair(image_id)
    
    # Format the filename exactly as evaluate.py saves it (replace / with _)
    safe_id = image_id.replace("/", "_") + ".png"
    canny_path = f"outputs/canny/{resolution}/{safe_id}"
    unet_path = f"outputs/unet/{resolution}/{safe_id}"
    
    # Load predictions (with fallbacks if Roles 2/3 haven't run yet)
    if os.path.exists(canny_path):
        canny_mask = cv2.imread(canny_path, cv2.IMREAD_GRAYSCALE) / 255.0
    else:
        canny_mask = np.zeros((512, 512)) # Black placeholder
        
    if os.path.exists(unet_path):
        unet_mask = cv2.imread(unet_path, cv2.IMREAD_GRAYSCALE) / 255.0
    else:
        unet_mask = np.zeros((512, 512)) # Black placeholder

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(color)
    axes[0].set_title("Color Input")
    axes[0].axis('off')
    
    axes[1].imshow(canny_mask, cmap='gray')
    axes[1].set_title("Canny Baseline")
    axes[1].axis('off')
    
    axes[2].imshow(unet_mask, cmap='gray')
    axes[2].set_title("U-Net Prediction")
    axes[2].axis('off')
    
    axes[3].imshow(true_mask, cmap='gray')
    axes[3].set_title("True Sketch")
    axes[3].axis('off')
    
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out_path = f"figures/{resolution}_{safe_id}"
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    import json
    with open("splits.json", "r") as f:
        splits = json.load(f)
    generate_report_figure(splits["train"][0])