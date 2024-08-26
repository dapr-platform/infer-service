import base64
import time

import cv2
import numpy as np
import subprocess
import threading
from typing import List, Tuple, Dict, Any
from util import logFunction, getPolygonFromConfiguration, get_absolute_polygon, getItemValueFromConfiguration
from event import Event, eventSender
import logging



class GrainOccupancyAnalysisHandler:
    def __init__(self, stop_channels: Dict[str, threading.Event]):
        self.LOW_VALUE = 10
        self.HIGH_VALUE = 30
        self.EVENT_THRESHOLD = 0.5
        self.EVENT_SUPRESS_SECONDS = 10
        self.stop_channels = stop_channels
        self.lock = threading.Lock()
        self.EVENT_TITLE = "grain_occupancy"
        self.EVENT_TEXT = "Grain Occupancy,当前值{current_value},门限值{event_threshold}"


    def get_info(self):
        return {
            "name": "Grain Occupancy Analysis",
            "configurations": [
                {"name": "low_value", "description": "颜色区间-低值", "default_value": 10, "type": "int"},
                {"name": "high_value", "description": "颜色区间-高值", "default_value": 30, "type": "int"},
                {"name": "event_threshold", "description": "触发事件阈值，低于触发", "default_value": 0.5,
                 "type": "float"},
                {"name": "event_supress_seconds", "description": "事件抑制秒数，限定秒数内只发送一次事件",
                 "default_value": 5, "type": "int"},
                {"name": "event_text","description": "事件文本", "default_value": "Grain Occupancy,当前值{current_value},门限值{event_threshold}", "type": "string"}
            ]
        }

    def process_img(self, img_base64: str,
                    configurations: List[Dict[str, any]] = None) -> Dict[str, Any]:
        """
        Processes a base64 encoded image, applies polygon mask, and performs grain occupancy analysis.

        Args:
            img_base64 (str): Base64 encoded image.

        Returns:
            Dict[str, Any]: Processed image with grain percentage information.
        """
        try:
            img_bytes = base64.b64decode(img_base64)
            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}")
        polygon = getPolygonFromConfiguration(configurations)
        if polygon and len(polygon) > 0:
            absPolygon = get_absolute_polygon(polygon, img.shape[1], img.shape[0])
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            points_array = np.array(absPolygon, dtype=np.int32)
            cv2.fillPoly(mask, [points_array], 255)
            masked_image = cv2.bitwise_and(img, img, mask=mask)
        else:
            masked_image = img

        blurred = cv2.GaussianBlur(masked_image, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        low_value = int(getItemValueFromConfiguration(configurations, 'low_value',
                                                      self.LOW_VALUE)) if configurations else self.LOW_VALUE
        high_value = int(getItemValueFromConfiguration(configurations, 'high_value',
                                                       self.HIGH_VALUE)) if configurations else self.HIGH_VALUE

        lower_yellow = np.array([low_value, 40, 40])
        upper_yellow = np.array([high_value, 255, 255])
        result = cv2.inRange(hsv, lower_yellow, upper_yellow)

        grain_pixels = np.sum(result == 255)
        region_pixels = np.sum(mask == 255) if polygon else img.size // 3
        grain_percentage = (grain_pixels / region_pixels) * 100 if region_pixels > 0 else 0

        overlay = img.copy()
        overlay[result == 255] = (0, 255, 255)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.putText(img, f'Grain: {grain_percentage:.2f}%', (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', img)
        result_img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {"result_img": result_img_base64, "grain_percentage": grain_percentage}

    def process_stream_worker(self, stream_url: str, id: str, out_stream_url: str,
                              configurations: List[Dict[str, any]] = None):
        """
        Processes a video stream, applies polygon mask, and performs grain occupancy analysis on each frame,
        then streams the result to an output stream using FFmpeg.

        Args:
            stream_url (str): URL of the video stream.
            id (str): Unique identifier for the stream processing.
            out_stream_url (str): URL of the output stream.

        Returns:
            None
        """
        logging.debug(f"Starting stream streamUrl={stream_url} outStreamUrl={out_stream_url}")
        cap = None
        process = None
        # 初始化抑制时间和最大值
        max_grain_percentage = 0
        last_event_time = time.time()

        sendedClearFlag = False

        try:
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                raise ValueError(f"Unable to open video stream: {stream_url}")

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            low_value = int(getItemValueFromConfiguration(configurations, 'low_value',
                                                          self.LOW_VALUE)) if configurations else self.LOW_VALUE
            high_value = int(getItemValueFromConfiguration(configurations, 'high_value',
                                                           self.HIGH_VALUE)) if configurations else self.HIGH_VALUE
            event_threshold = float(
                getItemValueFromConfiguration(configurations, 'event_threshold', self.EVENT_THRESHOLD)) if configurations else self.EVENT_THRESHOLD
            event_supress_seconds = int(getItemValueFromConfiguration(configurations, 'event_supress_seconds',self.EVENT_SUPRESS_SECONDS)) if configurations else self.EVENT_SUPRESS_SECONDS
            event_text = getItemValueFromConfiguration(configurations, 'event_text', self.EVENT_TEXT) if configurations else self.EVENT_TEXT

            polygon = getPolygonFromConfiguration(configurations)
            logging.debug(f"polygon={polygon}")
            absPolygon = None
            if polygon and len(polygon) > 0:
                absPolygon = get_absolute_polygon(polygon, frame_width, frame_height)
            logging.debug(f"absPolygon={absPolygon}")
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{frame_width}x{frame_height}',
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-g', str(fps),
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv',
                out_stream_url
            ]

            process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
            while True:
                if self.stop_channels.get(id) and self.stop_channels[id].is_set():
                    logFunction()
                    logging.error("Stopping stream")
                    break
                ret, frame = cap.read()
                if not ret:
                    logFunction()
                    logging.error(f"{stream_url} cap read error {ret}")
                    break
                if absPolygon is not None and len(absPolygon) > 0:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    points_array = np.array(absPolygon, dtype=np.int32)
                    cv2.fillPoly(mask, [points_array], 255)
                    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
                else:
                    masked_image = frame

                blurred = cv2.GaussianBlur(masked_image, (11, 11), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

                lower_yellow = np.array([low_value, 40, 40])
                upper_yellow = np.array([high_value, 255, 255])
                result = cv2.inRange(hsv, lower_yellow, upper_yellow)

                grain_pixels = np.sum(result == 255)
                region_pixels = np.sum(mask == 255) if polygon else frame.size // 3
                grain_percentage = (grain_pixels / region_pixels) if region_pixels > 0 else 0

                overlay = frame.copy()
                overlay[result == 255] = (0, 255, 255)
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                cv2.putText(frame, f'Grain: {grain_percentage:.2f}%', (10, frame_height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                process.stdin.write(frame.tobytes())
                process.stdin.flush()




                # 更新抑制时间内的最大值
                if grain_percentage > max_grain_percentage:
                    max_grain_percentage = grain_percentage

                # 发送一次清除事件
                if max_grain_percentage > event_threshold and not sendedClearFlag:
                    event = Event(id, self.EVENT_TITLE, event_text, time.time(), False, {"current_value": max_grain_percentage,"event_threshold":event_threshold})
                    eventSender.send_event(event)
                    sendedClearFlag = True  #只发送一次
                current_time = time.time()
                if (current_time - last_event_time) > event_supress_seconds:
                    # 发送抑制时间内的最大值事件
                    if max_grain_percentage < event_threshold:
                        event = Event(id, self.EVENT_TITLE, event_text, current_time,True,{ "current_value": max_grain_percentage,"event_threshold":event_threshold})
                        eventSender.send_event(event)
                        sendedClearFlag = False
                    # 重置事件时间和最大值
                    last_event_time = current_time
                    max_grain_percentage = 0


            process.stdin.close()
            process.wait()
        except Exception as e:
            logging.error(f"Error: {e}")
        finally:
            if process is not None:
                process.stdin.close()
                process.wait()
            if cap is not None:
                cap.release()
            with self.lock:
                if id in self.stop_channels:
                    del self.stop_channels[id]
