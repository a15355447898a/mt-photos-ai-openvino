#!/bin/bash
set -e

# 在启动主服务之前，先在单个进程中运行模型加载脚本。
# 这可以确保模型文件被安全地下载和解压，避免了多个工作进程同时下载造成的竞争条件。
# Gunicorn的 --preload 选项与此模式完美配合。
echo "--- Pre-loading models to avoid race conditions..."
python3 -c "from server import load_face_model, load_clip_img_model, load_clip_txt_model, load_ocr_model; load_face_model(); load_clip_img_model(); load_clip_txt_model(); load_ocr_model()"
echo "--- Model pre-loading complete. Starting supervisord..."

# 现在，用 exec 启动 supervisord
# supervisord 将成为容器的主进程 (PID 1)，并根据 /etc/supervisor/conf.d/supervisord.conf 的配置
# 来启动和管理 gunicorn 和 watchdog 两个服务。
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
