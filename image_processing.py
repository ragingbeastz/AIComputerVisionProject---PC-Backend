from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n-seg.pt")  # segmentation version

def process_image(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res = model(img)[0]

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for m, cls in zip(res.masks.xy, res.boxes.cls.cpu().numpy()):
        if int(cls) == 2:  # 'car'
            pts = np.int32([m])
            cv2.fillPoly(mask, pts, 255)

    white_bg = 255 * np.ones_like(img, dtype=np.uint8)
    final = np.where(mask[..., None] == 255, img, white_bg)

    h, w = final.shape[:2]

    scale = 300 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(final, (new_w, new_h))

    canvas = 255 * np.ones((300, 300, 3), dtype=np.uint8)

    x_off = (300 - new_w) // 2
    y_off = (300 - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    return canvas
