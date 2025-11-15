#!/bin/bash
set -e

# 在启动主服务之前，先在单个进程中运行模型加载脚本。
# 这可以确保模型文件被安全地下载和解压，避免了多个工作进程同时下载造成的竞争条件。
echo "--- Pre-loading models to avoid race conditions..."
python3 -c "from server import load_face_model, load_clip_img_model, load_clip_txt_model, load_ocr_model; load_face_model(); load_clip_img_model(); load_clip_txt_model(); load_ocr_model()"
echo "--- Model pre-loading complete. Starting Uvicorn server..."

# 现在，用 exec 启动主应用。
# exec 会让 Uvicorn 进程替换掉当前的 shell 进程，成为容器的主进程（PID 1），
# 这样可以更好地处理信号。
exec uvicorn server:app --host 0.0.0.0 --port ${HTTP_PORT:-8060} --workers ${WEB_CONCURRENCY:-1}
