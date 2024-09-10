import base64
import os
import torch
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
os.environ['TORCH_HOME'] = './torch_cache'
class YoloV5Test:
    def __init__(self, model_path: str):
        self.modelPath = model_path
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def process_img(self, img_base64: str):
        try:
            img_bytes = base64.b64decode(img_base64)
            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}")

        # YOLOv5 inference
        results = self.model(img)

        # Annotating the image
        annotated_img = np.squeeze(results.render())  # Rendered image with bounding boxes
        _, buffer = cv2.imencode('.jpg', annotated_img)
        result_img_base64 = base64.b64encode(buffer).decode('utf-8')

        predictions = []
        for pred in results.xyxy[0].tolist():  # xyxy format: [x_min, y_min, x_max, y_max, confidence, class_id]
            x_min, y_min, x_max, y_max, confidence, class_id = pred[:6]
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            confidence = float(confidence)
            class_id = int(class_id)
            predictions.append({
                "class_id": class_id,
                "class_name": self.model.names[class_id],
                "confidence": confidence,
                "box": [x_min, y_min, x_max, y_max]
            })

        return {"result_img": result_img_base64, "predictions": predictions}


if __name__ == "__main__":
    yoloTest = YoloV5Test("/Users/liu/Downloads/best.pt")
    print(yoloTest.process_img(YoloV5Test.encode_image("./bus.jpg")))