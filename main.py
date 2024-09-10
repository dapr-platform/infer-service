import base64
import os
import threading
import time
import subprocess
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

#from yolo_handle import YoloHandler
from yolo_multi_handle import YoloV5Handler, YoloV8Handler
from grain_occupancy_analysis_handle import GrainOccupancyAnalysisHandler
import logging
# 配置日志文件
logging.basicConfig(filename='main.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s')

# 获取环境变量 BASE_HOST 的值
base_host = os.getenv("BASE_HOST", "")

# 拼接前缀到 docs_url 和 openapi_url
openapi_url = f"{base_host}/swagger/doc.json"

print("BASE_HOST:", base_host)

# 初始化 FastAPI 应用
app = FastAPI(
    title="infer-service",
    description="AI inference service",
    version="1.0.0",
    docs_url="/swagger/index.html",
    openapi_url="/swagger/doc.json",
    swagger_ui_parameters={"url": f"{openapi_url}"},
    servers=[
        {"url": f"{base_host}", "description": "docker environment"},
        {"url": "/", "description": "local environment"},
    ]
)
# Globals
stop_channels = {}
executor = ThreadPoolExecutor(max_workers=10)
lock = threading.Lock()

# Initialize handlers
#yolo_handler = YoloHandler(stop_channels)
yolov5_handler = YoloV5Handler(stop_channels)
yolov8_handler = YoloV8Handler(stop_channels)
grain_occupancy_handler = GrainOccupancyAnalysisHandler(stop_channels)


class Result(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: List[int]


class Configuration(BaseModel):
    name: str
    value: Any


class ProcessImgRequest(BaseModel):
    img_base64: str
    model_name: str
    ai_type: str
    configurations: List[Dict[str, Any]]


class ProcessStreamRequest(BaseModel):
    stream_url: str
    id: str
    model_name: str
    ai_type: str
    out_stream_url: str
    configurations: List[Dict[str, Any]]


class ApiResponse(BaseModel):
    status: int
    msg: str
    data: Union[Dict[str, Any], List[Any], None] = None

@app.get("/list-running-streams", response_model=ApiResponse)
def list_running_streams():
    try:
        running_streams = list(stop_channels.keys())
        return ApiResponse(status=0, msg="Success", data=running_streams)
    except Exception as e:
        return ApiResponse(status=1, msg=str(e), data=None)
@app.get("/supported-ai-types", response_model=ApiResponse)
def get_supported_ai_types():
    response_data = {
        "machine_learning": [
            yolov5_handler.get_info(),
            yolov8_handler.get_info(),
            # Add other machine learning models here
        ],
        "opencv": [
            grain_occupancy_handler.get_info(),
            # Add other OpenCV models here
        ]
    }
    return ApiResponse(status=0, msg="Success", data=response_data)


@app.post("/process-image", response_model=ApiResponse)
def process_img(request: ProcessImgRequest):
    logging.debug(f"process image req={request}")
    ai_type = request.ai_type
    try:
        if ai_type == 'YOLOv5':
            result = yolov5_handler.process_img(request.img_base64, request.model_name,
                                              request.configurations)
        elif ai_type == 'YOLOv8':
            result = yolov8_handler.process_img(request.img_base64, request.model_name,
                                              request.configurations)
        elif ai_type == 'Grain Occupancy Analysis':
            result = grain_occupancy_handler.process_img(request.img_base64, request.configurations)
        else:
            raise HTTPException(status_code=400, detail="Unsupported ai_type")
        return ApiResponse(status=0, msg="Success", data=result)
    except Exception as e:
        return ApiResponse(status=1, msg=str(e), data=None)


@app.post("/process-stream", response_model=ApiResponse)
def process_stream(request: ProcessStreamRequest):
    logging.debug(f"process stream req={request}")
    id = request.id
    ai_type = request.ai_type

    if id in stop_channels:
        return ApiResponse(status=1, msg="Stream already running", data=None)

    stop_channels[id] = threading.Event()

    try:
        if ai_type == 'YOLO':
            executor.submit(yolo_handler.process_stream_worker, request.stream_url, id, request.model_name,
                            request.out_stream_url,  request.configurations)
        elif ai_type == 'Grain Occupancy Analysis':
            executor.submit(grain_occupancy_handler.process_stream_worker, request.stream_url, id,
                            request.out_stream_url,  request.configurations)
        else:
            raise HTTPException(status_code=400, detail="Unsupported ai_type")
        return ApiResponse(status=0, msg="Stream started", data=None)
    except Exception as e:
        return ApiResponse(status=1, msg=str(e), data=None)

@app.get("/check-running/{id}", response_model=ApiResponse)
def check_running(id: str):
    try:
        is_running = id in stop_channels and not stop_channels[id].is_set()
        if is_running:
            return ApiResponse(status=0, msg="Stream is running", data={"running": True})
        else:
            return ApiResponse(status=0, msg="Stream is not running", data={"running": False})
    except Exception as e:
        return ApiResponse(status=1, msg=str(e), data=None)

@app.post("/stop-stream", response_model=ApiResponse)
def stop_stream(id: str):
    if id in stop_channels:
        stop_channels[id].set()
        return ApiResponse(status=0, msg="Stream stopped", data=None)
    else:
        return ApiResponse(status=1, msg="Stream not running", data=None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
