# test_main.py
import base64
import pytest
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app

client = TestClient(app)

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def test_process_img_valid():
    img_base64 = encode_image("bus.jpg")
    response = client.post("/process-image", json={"img_base64": img_base64, "model_name": "yolov5x6u"})
    assert response.status_code == 200
    data = response.json()
    assert "result_img" in data
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    print(data["predictions"])

    result_img_base64 = data["result_img"]
    # 解码 base64 字符串为图像数据
    image_data = base64.b64decode(result_img_base64)
    # 将解码后的数据转换为图像
    image = Image.open(BytesIO(image_data))

    # 将图像保存到文件
    image.save("result_image.jpg", "JPEG")

def test_process_img_invalid_base64():
    response = client.post("/process_image", json={"img_base64": "invalid_base64", "model_name": "yolov5s"})
    assert response.status_code == 400
    assert response.json()["detail"].startswith("Failed to decode image")

def test_process_img_missing_model_name():
    img_base64 = encode_image("path/to/valid/image.jpg")
    response = client.post("/process_image", json={"img_base64": img_base64, "model_name": ""})
    assert response.status_code == 500
    assert "Model " in response.json()["detail"]


def test_process_stream_valid():
    response = client.post("/process_stream", json={
        "stream_url": "http://localhost:3000/api/stream/record/cam_video/test6.mp4.live.flv",
        "id": "test_stream_ai",
        "model_name": "yolov5nu",
        "out_stream_url": "rtmp://localhost:30006/live/testai"
    })
    assert response.status_code == 200
    assert response.json()["status"] == "started"

def test_process_stream_invalid_url():
    response = client.post("/process_stream", json={
        "stream_url": "http://example.com/invalid_stream",
        "id": "test_stream_2",
        "model_name": "yolov5s",
        "out_stream_url": "http://example.com/output_stream"
    })
    assert response.status_code == 500
    assert "Unable to open video stream" in response.json()["detail"]

def test_process_stream_missing_model_name():
    response = client.post("/process_stream", json={
        "stream_url": "http://localhost:3000/api/stream/record/cam_video/test6.mp4.live.flv",
        "id": "test_stream_3",
        "model_name": "",
        "out_stream_url": "rtmp://localhost/live/testai"
    })
    assert response.status_code == 500
    assert "Model " in response.json()["detail"]

def test_stop_stream_valid():
    response = client.post("/stop_stream", json={"id": "test_stream_1"})
    assert response.status_code == 200
    assert response.json()["status"] == "stopping"

def test_stop_stream_not_found():
    response = client.post("/stop_stream", json={"id": "non_existent_stream"})
    assert response.status_code == 200
    assert response.json()["status"] == "not found"