import os
import cv2
import numpy as np
from pathlib import Path
from rsslib.path import code_path

def create_combined_figure(step2_indices=[0, 2, 3, 5]):
    """
    Create a combined figure with white background:
    - Left: 2x2 grid from step1 (box, wall, shelf1, shelf2)
    - Right: 2x4 grid (top row: step2 images, bottom row: step3 images)
    
    Args:
        step2_indices: List of indices for step2/step3 images
    """
    base_path = Path(code_path) / "Visualization" / "paper" / "figure1"
    step1_path = base_path / "step1"
    step2_path = base_path / "step2"
    step3_path = base_path / "step3"
    
    # Load step1 images (2x2 grid)
    step1_images = [
        cv2.imread(str(step1_path / "box.png"), cv2.IMREAD_UNCHANGED),
        cv2.imread(str(step1_path / "wall.png"), cv2.IMREAD_UNCHANGED),
        cv2.imread(str(step1_path / "shelf1.png"), cv2.IMREAD_UNCHANGED),
        cv2.imread(str(step1_path / "shelf2.png"), cv2.IMREAD_UNCHANGED)
    ]
    
    # Check all step1 images loaded
    for i, img in enumerate(step1_images):
        if img is None:
            print(f"Warning: step1 image {i} not found")
            return
    
    # Convert RGBA to RGB with white background
    def rgba_to_rgb_white(img):
        if img.shape[2] == 4:  # Has alpha channel
            # Create white background
            bg = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            # Get alpha channel
            alpha = img[:, :, 3] / 255.0
            # Blend
            for c in range(3):
                bg[:, :, c] = (1 - alpha) * bg[:, :, c] + alpha * img[:, :, c]
            cv2.imwrite("test.png", bg)
            return bg
        return img
    
    step1_images = [rgba_to_rgb_white(img) for img in step1_images]
    
    # Resize all step1 images to same size
    h1, w1 = step1_images[0].shape[:2]
    step1_images = [cv2.resize(img, (w1, h1)) for img in step1_images]
    
    # Create 2x2 grid for step1
    top_row = np.hstack([step1_images[0], step1_images[1]])
    bottom_row = np.hstack([step1_images[2], step1_images[3]])
    step1_grid = np.vstack([top_row, bottom_row])
    
    # Load step2 images
    step2_images = []
    for idx in step2_indices:
        img = cv2.imread(str(step2_path / f"{idx}.png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: step2/{idx}.png not found")
            return
        step2_images.append(rgba_to_rgb_white(img))
    
    # Load step3 images
    step3_images = []
    for idx in step2_indices:
        img = cv2.imread(str(step3_path / f"{idx}.png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: step3/{idx}.png not found")
            return
        step3_images.append(rgba_to_rgb_white(img))
    
    # Resize step2/step3 images to same size
    h23, w23 = step2_images[0].shape[:2]
    step2_images = [cv2.resize(img, (w23, h23)) for img in step2_images]
    step3_images = [cv2.resize(img, (w23, h23)) for img in step3_images]
    
    # Create 2x4 grid for step2/step3
    step2_row = np.hstack(step2_images)  # Horizontal concatenation
    step3_row = np.hstack(step3_images)  # Horizontal concatenation
    step23_grid = np.vstack([step2_row, step3_row])  # Stack vertically
    
    # Resize step1_grid height to match step23_grid height
    step23_height = step23_grid.shape[0]
    step1_aspect = step1_grid.shape[1] / step1_grid.shape[0]
    new_step1_width = int(step23_height * step1_aspect)
    step1_grid_resized = cv2.resize(step1_grid, (new_step1_width, step23_height))
    
    # Concatenate horizontally: step1 on left, step23 on right
    final_figure = np.hstack([step1_grid_resized, step23_grid])
    
    # Save output
    output_path = base_path / "combined_figure.png"
    cv2.imwrite(str(output_path), final_figure)
    print(f"Saved combined figure to: {output_path}")
    print(f"Figure size: {final_figure.shape}")

if __name__ == "__main__":
    step2_indices = [0, 2, 3, 5]
    create_combined_figure(step2_indices)