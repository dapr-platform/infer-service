FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖和 Python 包
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libopencv-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
# 安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y net-tools git && rm -rf /var/lib/apt/lists/*


# 复制当前目录内容到容器的 /app 目录中
COPY . /app

# 暴露 FastAPI 默认端口
EXPOSE 90

# 运行 FastAPI 应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "90"]