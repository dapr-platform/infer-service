import base64
import os
import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from typing import List, Dict, Any, Tuple
import threading
import subprocess
import logging

from utils.augmentations import letterbox
from util import getPolygonFromConfiguration,get_absolute_polygon

class YoloHandler:
    def __init__(self, stop_channels: Dict[str, threading.Event]):
        self.img_net = {}
        self.stream_net = {}
        self.img_last_used = {}
        self.stream_last_used = {}
        self.stop_channels = stop_channels
        self.cache_duration = 10 * 60  # 10 minutes
        self.default_model_folder = "./ai_models"
        self.lock = threading.Lock()

    def get_info(self):
        return {
            "name": "YOLO",
            "configurations": [
                {"name": "confidence_threshold", "description": "门限", "default_value": 0.5,"type":"float"},
                {"name": "model_file", "description": "模型文件(新上传的文件名将以模型的名称+版本进行保存)", "default_value": "", "type": "file"},

            ]
        }
    def load_model(self, model_name: str):
        model_path = os.path.join(self.default_model_folder, f"{model_name}")
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_name} not found in {self.default_model_folder}")
        return YOLO(model_path)

    def get_img_net(self, model_name: str):
        with self.lock:
            if model_name in self.img_net and (time.time() - self.img_last_used[model_name]) < self.cache_duration:
                self.img_last_used[model_name] = time.time()
                return self.img_net[model_name]

            if model_name in self.img_net:
                del self.img_net[model_name]

            model = self.load_model(model_name)
            self.img_net[model_name] = model
            self.img_last_used[model_name] = time.time()
            return model

    def get_stream_net(self, model_name: str, id: str):
        with self.lock:
            net_key = f"{model_name}_{id}"
            if net_key in self.stream_net and (time.time() - self.stream_last_used[net_key]) < self.cache_duration:
                self.stream_last_used[net_key] = time.time()
                return self.stream_net[net_key]

            if net_key in self.stream_net:
                del self.stream_net[net_key]

            model = self.load_model(model_name)
            self.stream_net[net_key] = model
            self.stream_last_used[net_key] = time.time()
            return model

    def process_img(self, img_base64: str, model_name: str, configurations: List[Dict[str, any]] = None):
        try:
            img_bytes = base64.b64decode(img_base64)
            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}")
        polygon = getPolygonFromConfiguration(configurations)
        if polygon and len(polygon)>0:
            mask = np.zeros_like(img)
            absPolygon = get_absolute_polygon(polygon, img.shape[1], img.shape[0])

            cv2.fillPoly(mask, [np.array(absPolygon)], (255, 255, 255))
            img = cv2.bitwise_and(img, mask)

        confidence_threshold = float(configurations.get('confidence_threshold', 0.5)) if configurations else 0.5
        model = self.get_img_net(model_name)
        results = model(img)

        annotated_img = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_img)
        result_img_base64 = base64.b64encode(buffer).decode('utf-8')

        predictions = []
        for pred in results[0].boxes.data.tolist():
            x_min, y_min, x_max, y_max, confidence, class_id = pred[:6]
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            confidence = float(confidence)
            class_id = int(class_id)
            predictions.append({
                "class_id": class_id,
                "class_name": results[0].names[class_id],
                "confidence": confidence,
                "box": [x_min, y_min, x_max, y_max]
            })

        return {"result_img": result_img_base64, "predictions": predictions}

    def process_stream_worker(self, stream_url: str, id: str, model_name: str, out_stream_url: str, configurations: List[Dict[str, any]] = None):
        process = None
        cap = None
        try:
            model = self.get_stream_net(model_name, id)
            input_size = model.model.args["imgsz"]
            WIDTH, HEIGHT, FPS = self.get_video_dimensions(stream_url)

            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                raise ValueError(f"Unable to open video stream: {stream_url}")

            ffmpeg_cmd = ['ffmpeg',
                          '-y',
                          '-f', 'rawvideo',
                          '-vcodec', 'rawvideo',
                          '-pix_fmt', 'bgr24',
                          '-s', f'{WIDTH}x{HEIGHT}',
                          '-r', f'{FPS}',
                          '-i', '-',
                          '-c:v', 'libx264',
                          '-g', f'{FPS}',
                          '-pix_fmt', 'yuv420p',
                          '-preset', 'ultrafast',
                          '-f', 'flv',
                          out_stream_url]

            process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
            polygon = getPolygonFromConfiguration(configurations)
            absPolygon = None
            if polygon and len(polygon) > 0:
                absPolygon = get_absolute_polygon(polygon, WIDTH, HEIGHT)

            while True:
                if self.stop_channels[id].is_set():
                    break
                ret_val, img0 = cap.read()
                if not ret_val:
                    logging.error(f"Failed to read frame from video stream: {stream_url}")
                    break

                if absPolygon is not None and len(absPolygon) > 0:
                    mask = np.zeros_like(img0)
                    cv2.fillPoly(mask, [np.array(absPolygon)], (255, 255, 255))
                    img0 = cv2.bitwise_and(img0, mask)

                #img = self.preprocess_image(img0, input_size)
                #logging.debug(f"Preprocessed image shape: {img.shape}")
                try:
                    results = model.predict(source=img0,imgsz=input_size)
                except Exception as e:
                    #logging.error(f"Exception during model inference: {e}", exc_info=True)
                    process.stdin.write(img0.tobytes())
                    process.stdin.flush()
                    continue

                annotated_img = results[0].plot()
                if annotated_img is None or annotated_img.size == 0:
                    process.stdin.write(img0.tobytes())
                    process.stdin.flush()
                    continue

                bgr_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                if bgr_img.shape[1] != WIDTH or bgr_img.shape[0] != HEIGHT:
                    bgr_img = cv2.resize(bgr_img, (WIDTH, HEIGHT))

                process.stdin.write(bgr_img.tobytes())
                process.stdin.flush()

            process.stdin.close()
            process.wait()
        except Exception as e:
            print(f"Error: {e}")
            logging.error(f"Error: {e}", exc_info=True)
        finally:
            if process is not None:
                process.stdin.close()
                process.wait()
            if cap is not None:
                cap.release()
            with self.lock:
                if id in self.stop_channels:
                    del self.stop_channels[id]


    @staticmethod
    def preprocess_image(img0, input_size):
        img = letterbox(img0, input_size, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return img

    @staticmethod
    def get_video_dimensions(stream_url: str):
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video stream: {stream_url}")
        WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = int(cap.get(cv2.CAP_PROP_FPS))
        logging.debug(f"Video dimensions: {WIDTH}x{HEIGHT}, FPS: {FPS}")
        cap.release()
        return WIDTH, HEIGHT, FPS