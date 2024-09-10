import base64
import os
import time
import torch
import cv2
import numpy as np
from typing import List, Dict, Any
import threading
import subprocess
import logging

os.environ['TORCH_HOME'] = './torch_cache'
from utils.augmentations import letterbox
from util import getPolygonFromConfiguration, get_absolute_polygon


class YoloBaseHandler:
    def __init__(self, stop_channels: Dict[str, threading.Event]):
        self.img_models = {}
        self.stream_models = {}
        self.img_last_used = {}
        self.stream_last_used = {}
        self.stop_channels = stop_channels
        self.cache_duration = 10 * 60  # 10 minutes
        self.default_model_folder = "./ai_models"
        self.lock = threading.Lock()
        self.configurations = [
            {"name": "confidence_threshold", "description": "门限", "default_value": 0.5, "type": "float"},
            {"name": "model_file", "description": "模型文件(新上传的文件名将以模型的名称+版本进行保存)",
             "default_value": "", "type": "file"},
        ]
        threading.Thread(target=self.clean_cache, daemon=True).start()

    def load_model(self, model_name: str):
        model_path = os.path.join(self.default_model_folder, f"{model_name}")
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_name} not found in {self.default_model_folder}")
        return self.load_specific_model(model_path)  # 调用子类实现

    def get_cached_model(self, model_name: str):
        with self.lock:
            if model_name in self.img_models and (time.time() - self.img_last_used[model_name]) < self.cache_duration:
                self.img_last_used[model_name] = time.time()
                return self.img_models[model_name]

            if model_name in self.img_models:
                del self.img_models[model_name]

            model = self.load_model(model_name)
            self.img_models[model_name] = model
            self.img_last_used[model_name] = time.time()
            return model

    def get_stream_cached_model(self, model_name: str, id: str):
        with self.lock:
            net_key = f"{model_name}_{id}"
            if net_key in self.stream_models and (time.time() - self.stream_last_used[net_key]) < self.cache_duration:
                self.stream_last_used[net_key] = time.time()
                return self.stream_models[net_key]

            if net_key in self.stream_models:
                del self.stream_models[net_key]

            model = self.load_model(model_name)
            self.stream_models[net_key] = model
            self.stream_last_used[net_key] = time.time()
            return model

    def clean_cache(self):
        while True:
            with self.lock:
                current_time = time.time()
                self.img_models = {k: v for k, v in self.img_models.items() if
                                   (current_time - self.img_last_used[k]) < self.cache_duration}
                self.stream_models = {k: v for k, v in self.stream_models.items() if
                                      (current_time - self.stream_last_used[k]) < self.cache_duration}
            time.sleep(60)  # 每60秒检查一次
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

    def process_img(self, img_base64: str, model_name: str, configurations: List[Dict[str, any]] = None):
        try:
            if ',' in img_base64:
                img_base64 = img_base64.split(',', 1)[1]
            img_bytes = base64.b64decode(img_base64)
            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Image decoding failed, possibly due to invalid image format.")
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}")

        polygon = getPolygonFromConfiguration(configurations)
        if polygon and len(polygon) > 0:
            mask = np.zeros_like(img)
            absPolygon = get_absolute_polygon(polygon, img.shape[1], img.shape[0])
            cv2.fillPoly(mask, [np.array(absPolygon)], (255, 255, 255))
            img = cv2.bitwise_and(img, mask)

        confidence_threshold = 0.5
        if configurations:
            for config in configurations:
                if config["name"] == "confidence_threshold":
                    confidence_threshold = float(config["default_value"])
        model = self.get_cached_model(model_name)

        return self.run_inference(model, img, confidence_threshold)

    def process_stream_worker(self, stream_url: str, id: str, model_name: str, out_stream_url: str,
                              configurations: List[Dict[str, any]] = None):
        process = None
        cap = None
        try:
            model = self.get_stream_cached_model(model_name, id)
            input_size = self.get_input_size(model)
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

                ### update last use time
                model = self.get_stream_cached_model(model_name, id)
                ret_val, img0 = cap.read()
                if not ret_val:
                    logging.error(f"Failed to read frame from video stream: {stream_url}")
                    break

                if absPolygon is not None and len(absPolygon) > 0:
                    mask = np.zeros_like(img0)
                    cv2.fillPoly(mask, [np.array(absPolygon)], (255, 255, 255))
                    img0 = cv2.bitwise_and(img0, mask)

                try:
                    results = self.run_stream_inference(model, img0, input_size)
                except Exception as e:
                    logging.error(f"Exception during model inference: {e}", exc_info=True)
                    process.stdin.write(img0.tobytes())
                    process.stdin.flush()
                    continue

                annotated_img = self.get_annotated_img(results)
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

    # 子类需要实现的方法
    def get_info(self):
        raise NotImplementedError

    def load_specific_model(self, model_path: str):
        raise NotImplementedError

    def run_inference(self, model, img, confidence_threshold: float):
        raise NotImplementedError

    def run_stream_inference(self, model, img, input_size):
        raise NotImplementedError

    def get_annotated_img(self, results):
        raise NotImplementedError

    def get_input_size(self, model):
        raise NotImplementedError


# YOLOv5 实现类
class YoloV5Handler(YoloBaseHandler):
    def get_info(self):
        return {
            "name": "YOLOv5",
            "configurations": self.configurations
        }

    def load_specific_model(self, model_path: str):
        # 使用 torch.hub.load 来加载 YOLOv5 模型
        return torch.hub.load('torch_cache/hub/ultralytics_yolov5_master', 'custom', path=model_path,source='local')

    def run_inference(self, model, img, confidence_threshold: float):
        results = model(img)

        # YOLOv5 结果处理
        annotated_img = np.squeeze(results.render())
        _, buffer = cv2.imencode('.jpg', annotated_img)
        result_img_base64 = base64.b64encode(buffer).decode('utf-8')

        predictions = []
        for pred in results.xyxy[0].tolist():
            x_min, y_min, x_max, y_max, confidence, class_id = pred[:6]
            if confidence < confidence_threshold:
                continue
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            confidence = float(confidence)
            class_id = int(class_id)
            predictions.append({
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": confidence,
                "box": [x_min, y_min, x_max, y_max]
            })

        return {"result_img": result_img_base64, "predictions": predictions}

    def run_stream_inference(self, model, img, input_size):

        # YOLOv5 推理时需要对输入图像进行 letterbox 预处理，并进行标准化
        #img = letterbox(img, input_size, stride=model.stride.max(), auto=True)[0]
        #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        #img = np.ascontiguousarray(img)

        #img = torch.from_numpy(img).to(model.device)
        #img = img.float()  # Convert to float32
        #img /= 255.0  # Normalize to [0, 1]

        #if img.ndimension() == 3:
        #    img = img.unsqueeze(0)  # Add batch dimension

        #with torch.no_grad():
        #    results = model(img)

        #return results
        return model(img)

    def get_annotated_img(self, results):
        return np.squeeze(results.render())

    def get_input_size(self, model):
        if hasattr(model, 'imgsz'):
            return model.imgsz
        return (640, 640)  # 默认输入尺寸


# YOLOv8 实现类
class YoloV8Handler(YoloBaseHandler):
    def get_info(self):
        return {
            "name": "YOLOv8",
            "configurations": self.configurations
        }

    def load_specific_model(self, model_path: str):
        # 使用 ultralytics 库来加载 YOLOv8 模型
        from ultralytics import YOLO
        return YOLO(model_path)

    def run_inference(self, model, img, confidence_threshold: float):
        results = model(img)

        # YOLOv8 结果处理
        annotated_img = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_img)
        result_img_base64 = base64.b64encode(buffer).decode('utf-8')

        predictions = []
        for pred in results[0].boxes.data.tolist():
            x_min, y_min, x_max, y_max, confidence, class_id = pred[:6]
            if confidence < confidence_threshold:
                continue
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

    def run_stream_inference(self, model, img, input_size):
        return model.predict(source=img, imgsz=input_size)

    def get_annotated_img(self, results):
        return results[0].plot()

    def get_input_size(self, model):
        return model.model.args.get("imgsz", 640)
