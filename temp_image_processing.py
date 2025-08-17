from ultralytics import YOLO
import cv2
import numpy as np

def process_image(image):
    image = np.array(image)
    h, w = image.shape[:2]

    scale = 299 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = 255 * np.ones((299, 299, 3), dtype=np.uint8)

    x_off = (299 - new_w) // 2
    y_off = (299 - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    return canvas
