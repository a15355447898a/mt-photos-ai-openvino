# MT Photos AI识别相关任务独立部署项目

合并了 https://github.com/MT-Photos/mt-photos-ai 和 https://github.com/kqstone/mt-photos-insightface-unofficial 部分的代码

运行一个程序就可以支持以上2个程序的功能

> 适用于 Intel Arc GPU

## 最近更新

- **引入了完善的多线程支持**：重构了服务架构，现在所有AI接口都能稳定、高效地并行处理多个请求。
- **提升了可配置性**：新增了 `WEB_CONCURRENCY` 环境变量，允许用户根据服务器性能自由调整并发的工作进程数量。同时，OCR任务的计算设备（`OCR_DEVICE`）和内部并发数（`OCR_INFER_REQUESTS`）也可通过环境变量灵活切换。

### OCR 模式切换说明

使用下方环境变量控制识别分支：

```bash
- OCR_REC_DYNAMIC_WIDTH=on    # on=动态宽度，off=静态宽度
- OCR_DEVICE=CPU              # CPU 或 GPU
```

实测表现（以Arc独显为例）：

| 设备+模式        | 速度 | 精度表现 |
|-----------------|------|----------|
| GPU + 静态宽度  | 非常快 | 个别不存在的字会误检 |
| GPU + 动态宽度  | 非常慢 | 精度略高，误检减少 |
| CPU + 动态宽度  | 较快  | 精度最高，长文本也能稳定识别 |
| CPU + 静态宽度  | 略快于上者 | 精度与 CPU+动态 持平 |

## 配置选项 (环境变量)

你可以在 `docker run` 命令中使用 `-e` 参数来配置服务：

| 环境变量 | 功能说明 | 默认值 |
| :--- | :--- | :--- |
| `API_AUTH_KEY` | 用于访问API的认证密钥。 | `mt_photos_ai_extra` |
| `WEB_CONCURRENCY` | Uvicorn服务启动的工作进程数，推荐设置为CPU核心数。 | `4` |
| `OCR_DEVICE` | 指定OCR任务使用的计算设备。 | `CPU` |
| `OCR_INFER_REQUESTS`| 每个工作进程为OCR创建的并行推理请求数。 | `2` |
| `HTTP_PORT` | 容器内服务监听的端口号。 | `8060` |


## Docker Compose 部署指南

### 1. 准备工作：下载CLIP模型

在构建镜像前，你需要手动下载两个CLIP模型文件，并将它们放置在 `openvino/utils/` 目录下。

- `vit-b-16.img.fp32.onnx`
- `vit-b-16.txt.fp32.onnx`

**下载链接**: https://github.com/MT-Photos/mt-photos-ai/releases/tag/v1.1.0

### 2. 创建并运行服务

在项目根目录下（与 `openvino` 目录同级），创建一个名为 `docker-compose.yml` 的文件，并将以下内容复制进去。

```yaml
version: '3.8'

services:
  mt-photos-ai:
    image: mt-photos-ai:openvino
    build:
      context: ./openvino
      dockerfile: Dockerfile
    container_name: mt-photos-ai
    ports:
      - "8060:8060"
    devices:
      - "/dev/dri:/dev/dri"
    environment:
      - API_AUTH_KEY=your_secret_key_here
      - WEB_CONCURRENCY=4
      - OCR_DEVICE=GPU
      - OCR_INFER_REQUESTS=2
      - HTTP_PORT=8060
    restart: unless-stopped
```

然后，在 `docker-compose.yml` 文件所在的目录中，执行以下命令来一次性完成构建和启动：

```bash
docker-compose up --build -d
```
- `--build` 参数会确保在启动前（重新）构建镜像。
- `-d` 参数使服务在后台运行。


看到以下日志，则说明服务已经启动成功
```bash
INFO:     Started server process [3024]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8060 (Press CTRL+C to quit)
```


## API

### /check

检测服务是否可用，及api-key是否正确

```bash
curl --location --request POST 'http://127.0.0.1:8000/check' \
--header 'api-key: api_key'
```

**response:**

```json
{
  "result": "pass"
}
```

### /ocr

```bash
curl --location --request POST 'http://127.0.0.1:8000/ocr' \
--header 'api-key: api_key' \
--form 'file=@"/path_to_file/test.jpg"'
```

**response:**

- result.texts : 识别到的文本列表
- result.scores : 为识别到的文本对应的置信度分数，1为100%
- result.boxes : 识别到的文本位置，x,y为左上角坐标，width,height为框的宽高

```json
{
  "result": {
    "texts": [
      "识别到的文本1",
      "识别到的文本2"
    ],
    "scores": [
      "0.98",
      "0.97"
    ],
    "boxes": [
      {
        "x": "4.0",
        "y": "7.0",
        "width": "283.0",
        "height": "21.0"
      },
      {
        "x": "7.0",
        "y": "34.0",
        "width": "157.0",
        "height": "23.0"
      }
    ]
  }
}
```


### /clip/img

```bash
curl --location --request POST 'http://127.0.0.1:8000/clip/img' \
--header 'api-key: api_key' \
--form 'file=@"/path_to_file/test.jpg"'
```

**response:**

- results : 图片的特征向量

```json
{
  "results": [
    "0.3305919170379639",
    "-0.4954293668270111",
    "0.0217289477586746",
    ...
  ]
}
```

### /clip/txt

```bash
curl --location --request POST 'http://127.0.0.1:8000/clip/txt' \
--header "Content-Type: application/json" \
--header 'api-key: api_key' \
--data '{"text":"飞机"}'
```

**response:**

- results : 文字的特征向量

```json
{
  "results": [
    "0.3305919170379639",
    "-0.4954293668270111",
    "0.0217289477586746",
    ...
  ]
}
```

### /restart_v2

通过重启进程来释放内存

```bash
curl --location --request POST 'http://127.0.0.1:8000/restart_v2' \
--header 'api-key: api_key'
```

**response:**

请求中断,没有返回，因为服务重启了
