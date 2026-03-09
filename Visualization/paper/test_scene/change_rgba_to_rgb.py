import cv2
import numpy as np

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

img = cv2.imread("/home/mingi/RSS_2026/Visualization/paper/test_scene/capture_20260129_160616.png", cv2.IMREAD_UNCHANGED)
rgb_img = rgba_to_rgb_white(img)
cv2.imwrite("/home/mingi/RSS_2026/Visualization/paper/test_scene/capture_20260129_160616_rgb.png", rgb_img)