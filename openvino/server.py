from dotenv import load_dotenv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import cv2
import asyncio
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import insightface
from insightface.utils import storage
from insightface.app import FaceAnalysis
import logging
import utils.clip as clip
import openvino as ov
from pathlib import Path
import copy
import math
import utils.ocr_pre_post_processing as ocr_processing
from queue import Queue

logging.basicConfig(level=logging.WARNING)


# import onnxruntime as ort
# device = ort.get_device()
# print(f"Using device: {device}")

on_linux = sys.platform.startswith('linux')

load_dotenv()
app = FastAPI()
api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = int(os.getenv("HTTP_PORT", "8060"))
server_restart_time = int(os.getenv("SERVER_RESTART_TIME", "300"))
env_auto_load_txt_modal = os.getenv("AUTO_LOAD_TXT_MODAL", "off") == "on" # 是否自动加载CLIP文本模型，开启可以优化第一次搜索时的响应速度,文本模型占用700多m内存

restart_timer = None
clip_img_model = None
clip_txt_model = None
clip_img_request_pool = None
clip_txt_request_pool = None
face_model_pool = None
models_warmed = False

# OCR settings
OCR_INFER_REQUESTS = int(os.getenv("OCR_INFER_REQUESTS", "8"))  # Number of parallel inference requests for OCR
ocr_device = os.getenv("OCR_DEVICE", "CPU")  # CPU or GPU
det_model_file_path = Path("model/ch_PP-OCRv4_det_infer/inference.pdmodel")
rec_model_file_path = Path("model/ch_PP-OCRv4_rec_infer/inference.pdmodel")
ocr_rec_dynamic_width = os.getenv("OCR_REC_DYNAMIC_WIDTH", "on") == "on"
ov_core = None
det_compiled_model = None
rec_compiled_model = None
det_request_pool = None
rec_request_pool = None
postprocess_op = None

# Face Recognition settings
FACE_PARALLEL_INSTANCES = int(os.getenv("FACE_PARALLEL_INSTANCES", "2"))

# Thread pool settings
clip_workers = int(os.getenv("CLIP_WORKERS", "4"))
ocr_workers = int(os.getenv("OCR_WORKERS", str(OCR_INFER_REQUESTS)))
face_workers = int(os.getenv("FACE_WORKERS", str(FACE_PARALLEL_INSTANCES)))
clip_img_infer_requests = max(1, int(os.getenv("CLIP_IMG_INFER_REQUESTS", str(clip_workers))))
clip_txt_infer_requests = max(1, int(os.getenv("CLIP_TXT_INFER_REQUESTS", str(clip_workers))))

ocr_executor = ThreadPoolExecutor(max_workers=ocr_workers)
clip_executor = ThreadPoolExecutor(max_workers=clip_workers)
face_executor = ThreadPoolExecutor(max_workers=face_workers)


face_analysis_device = "GPU" #值：GPU || CPU ,指定使用核显还是CPU识别,默认GPU， 创建容器时，需要 --device /dev/dri:/dev/dri
detector_backend = os.getenv("DETECTOR_BACKEND", "insightface")
recognition_model = os.getenv("RECOGNITION_MODEL", "buffalo_l")
detection_thresh = float(os.getenv("DETECTION_THRESH", "0.65"))
# 设置下载模型URL
storage.BASE_REPO_URL = 'https://github.com/kqstone/mt-photos-insightface-unofficial/releases/download/models'
on_win = sys.platform.startswith('win')
model_folder_path = '~/.insightface'
if on_win :
    current_folder = os.path.dirname(os.path.abspath(__file__))
    model_folder_path = os.path.join(current_folder, "_insightface_root")


class ClipTxtRequest(BaseModel):
    text: str

def load_ocr_model():
    global ov_core, det_compiled_model, rec_compiled_model, postprocess_op, det_request_pool, rec_request_pool
    if ov_core is None:
        print(f"\n[INFO] Initializing OpenVINO OCR on device: {ocr_device}")
        ov_core = ov.Core()

        # Text detection model
        print(f"[INFO] Loading detection model: {det_model_file_path}")
        det_model = ov_core.read_model(det_model_file_path)
        det_compiled_model = ov_core.compile_model(det_model, device_name=ocr_device)

        # Create a pool of inference requests for the detection model
        det_request_pool = Queue(maxsize=OCR_INFER_REQUESTS)
        for _ in range(OCR_INFER_REQUESTS):
            det_request_pool.put(det_compiled_model.create_infer_request())

        # Text recognition model (dynamic width)
        print(f"[INFO] Loading recognition model: {rec_model_file_path}")
        rec_model = ov_core.read_model(rec_model_file_path)
        if ocr_rec_dynamic_width:
            for input_layer in rec_model.inputs:
                shape = input_layer.partial_shape
                shape[3] = -1  # Set width to dynamic
                rec_model.reshape({input_layer: shape})

        rec_compiled_model = ov_core.compile_model(rec_model, device_name=ocr_device)

        # Create a pool of inference requests for the recognition model
        rec_request_pool = Queue(maxsize=OCR_INFER_REQUESTS)
        for _ in range(OCR_INFER_REQUESTS):
            rec_request_pool.put(rec_compiled_model.create_infer_request())

        # Post-processing operator
        ocr_processing.postprocess_params["character_dict_path"] = "fonts/ppocr_keys_v1.txt"
        postprocess_op = ocr_processing.build_post_process(ocr_processing.postprocess_params)
        print("[INFO] OCR models and request pools loaded successfully.")


def load_clip_img_model():
    global clip_img_model, clip_img_request_pool
    if clip_img_model is None:
        clip_img_model = clip.load_img_model()
        clip_img_request_pool = Queue(maxsize=clip_img_infer_requests)
        for _ in range(clip_img_infer_requests):
            clip_img_request_pool.put(clip_img_model.create_infer_request())

def load_clip_txt_model():
    global clip_txt_model, clip_txt_request_pool
    if clip_txt_model is None:
        clip_txt_model = clip.load_txt_model()
        clip_txt_request_pool = Queue(maxsize=clip_txt_infer_requests)
        for _ in range(clip_txt_infer_requests):
            clip_txt_request_pool.put(clip_txt_model.create_infer_request())

def load_face_model():
    global face_model_pool
    if face_model_pool is None:
        print(f"\n[INFO] Initializing {FACE_PARALLEL_INSTANCES} face model instances...")
        face_model_pool = Queue(maxsize=FACE_PARALLEL_INSTANCES)
        for _ in range(FACE_PARALLEL_INSTANCES):
            faceAnalysis = FaceAnalysis(
                providers=["OpenVINOExecutionProvider"],
                provider_options=[{"device_type": face_analysis_device, "precision": "FP32"}],
                root=model_folder_path,
                allowed_modules=['detection', 'recognition'],
                name=recognition_model
            )
            faceAnalysis.prepare(ctx_id=0, det_thresh=detection_thresh, det_size=(640, 640))
            face_model_pool.put(faceAnalysis)
        print("[INFO] Face model pool loaded successfully.")


@app.on_event("startup")
async def startup_event():
    # Load models on startup to avoid cold starts on first request
    load_face_model()
    load_clip_img_model()
    load_ocr_model()
    load_clip_txt_model()
    await warmup_models()


@app.middleware("http")
async def check_activity(request, call_next):
    global restart_timer

    if restart_timer:
        restart_timer.cancel()

    restart_timer = threading.Timer(server_restart_time, restart_program)
    restart_timer.start()

    response = await call_next(request)
    return response

async def verify_header(api_key: str = Header(...)):
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

async def predict(predict_func, *args, executor=None):
    """Helper to run blocking functions in dedicated thread pools."""
    loop = asyncio.get_running_loop()
    func = partial(predict_func, *args)
    return await loop.run_in_executor(executor, func)

# --- OCR Helper Functions ---

def _ocr_image_preprocess(input_image, size=640):
    img = cv2.resize(input_image, (size, size))
    img = np.transpose(img, [2, 0, 1]) / 255.0
    img = np.expand_dims(img, 0)
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = (img - img_mean) / img_std
    return img.astype(np.float32)

def _ocr_resize_norm_img(img, max_wh_ratio):
    rec_image_shape = [3, 48, 320]
    imgC, imgH, imgW = rec_image_shape
    assert imgC == img.shape[2]

    if ocr_rec_dynamic_width and "ch" == "ch":
        imgW = int(32 * max_wh_ratio)

    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))

    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype("float32")
    resized_image = resized_image.transpose((2, 0, 1)) / 255.0
    resized_image = (resized_image - 0.5) / 0.5

    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im

def _ocr_post_processing_detection(frame, det_results):
    ori_im = frame.copy()
    data = {"image": frame}
    data_resize = ocr_processing.DetResizeForTest(data)
    img, shape_list = data_resize["image"], data_resize["shape"]
    shape_list = np.expand_dims(shape_list, axis=0)

    pred = det_results[0]
    segmentation = pred > 0.3
    boxes_batch = []
    for batch_index in range(pred.shape[0]):
        src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
        mask = segmentation[batch_index]
        boxes, scores = ocr_processing.boxes_from_bitmap(
            pred[batch_index], mask, src_w, src_h
        )
        boxes_batch.append({"points": boxes})

    post_result = boxes_batch
    dt_boxes = post_result[0]["points"]
    dt_boxes = ocr_processing.filter_tag_det_res(dt_boxes, ori_im.shape)
    return dt_boxes

def _ocr_batch_text_box(img_crop_list, img_num, indices, beg_img_no, batch_num):
    norm_img_batch = []
    max_wh_ratio = 0
    end_img_no = min(img_num, beg_img_no + batch_num)

    for ino in range(beg_img_no, end_img_no):
        h, w = img_crop_list[indices[ino]].shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)

    for ino in range(beg_img_no, end_img_no):
        norm_img = _ocr_resize_norm_img(img_crop_list[indices[ino]], max_wh_ratio)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch.append(norm_img)

    norm_img_batch = np.concatenate(norm_img_batch)
    return norm_img_batch.copy()

def _run_ocr_pipeline(image_bytes):
    """Synchronous helper function containing the full OCR pipeline."""
    det_request = None
    rec_request = None
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {'result': [], 'msg': 'Cannot decode image'}

        height, width, _ = frame.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}

        # --- Start OCR Pipeline ---
        test_image = _ocr_image_preprocess(frame, size=640)

        # Get a detection request from the pool and perform inference
        det_request = det_request_pool.get()
        det_results = det_request.infer([test_image])
        det_request_pool.put(det_request)
        det_request = None # Clear reference

        dt_boxes = _ocr_post_processing_detection(frame, det_results[det_compiled_model.output(0)])

        if dt_boxes is None or len(dt_boxes) == 0:
            return {"result": {"texts": [], "scores": [], "boxes": []}}

        dt_boxes = ocr_processing.sorted_boxes(dt_boxes)

        img_crop_list = [ocr_processing.get_rotate_crop_image(frame, box) for box in dt_boxes]

        img_num = len(img_crop_list)
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_crop_list]
        indices = np.argsort(np.array(width_list))

        batch_num = 6
        rec_res = [["", 0.0]] * img_num
        for beg_img_no in range(0, img_num, batch_num):
            norm_img_batch = _ocr_batch_text_box(
                img_crop_list, img_num, indices, beg_img_no, batch_num
            )

            # Get a recognition request from the pool and perform inference
            rec_request = rec_request_pool.get()
            rec_results = rec_request.infer([norm_img_batch])
            rec_request_pool.put(rec_request)
            rec_request = None # Clear reference

            rec_result = postprocess_op(rec_results[rec_compiled_model.output(0)])
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        texts = [str(r[0]) for r in rec_res]
        scores = [f"{r[1]:.2f}" for r in rec_res]

        boxes_xywh = []
        for box in dt_boxes:
            box = np.array(box, dtype=np.float32)
            x_min = float(np.min(box[:, 0]))
            y_min = float(np.min(box[:, 1]))
            width = float(np.max(box[:, 0]) - x_min)
            height = float(np.max(box[:, 1]) - y_min)
            boxes_xywh.append({
                'x': f"{x_min:.1f}", 'y': f"{y_min:.1f}",
                'width': f"{width:.1f}", 'height': f"{height:.1f}",
            })

        final_result = {"texts": texts, "scores": scores, "boxes": boxes_xywh}
        return {'result': final_result}
    except Exception as e:
        logging.error("OCR Error: %s", e, exc_info=True)
        return {'result': [], 'msg': str(e)}
    finally:
        # Ensure requests are returned to the pool even if an error occurs
        if det_request is not None:
            det_request_pool.put(det_request)
        if rec_request is not None:
            rec_request_pool.put(rec_request)

def _decode_upload_image(image_bytes, content_type=None, filename=""):
    """Decode upload bytes into a BGR OpenCV image."""
    img = None
    try:
        if content_type == 'image/gif':
            with Image.open(BytesIO(image_bytes)) as img_pil:
                if img_pil.is_animated:
                    img_pil.seek(0)
                frame = img_pil.convert('RGB')
                np_arr = np.array(frame)
                img = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
        if img is None:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        img = None

    if img is None:
        label = filename or "uploaded file"
        raise ValueError(f"The uploaded file {label} is not a valid image format or is corrupted.")
    return img

def _clip_img_pipeline(image_bytes):
    try:
        img = _decode_upload_image(image_bytes)
        infer_request = clip_img_request_pool.get()
        try:
            result = clip.process_image(img, clip_img_model, infer_request=infer_request)
        finally:
            clip_img_request_pool.put(infer_request)
        return {'result': ["{:.16f}".format(vec) for vec in result]}
    except Exception as e:
        logging.error("CLIP image error: %s", e, exc_info=True)
        return {'result': [], 'msg': str(e)}

def _clip_txt_pipeline(text):
    try:
        infer_request = clip_txt_request_pool.get()
        try:
            result = clip.process_txt(text, clip_txt_model, infer_request=infer_request)
        finally:
            clip_txt_request_pool.put(infer_request)
        return {'result': ["{:.16f}".format(vec) for vec in result]}
    except Exception as e:
        logging.error("CLIP text error: %s", e, exc_info=True)
        return {'result': [], 'msg': str(e)}

def _run_represent_pipeline(image_bytes, content_type, filename):
    try:
        img = _decode_upload_image(image_bytes, content_type, filename)
        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}

        embedding_objs = _represent(img)
        data = {"detector_backend": detector_backend, "recognition_model": recognition_model, "result": embedding_objs}
        logging.info("detected_img: %s, detected_persons: %d", filename or "unknown", len(embedding_objs))
        return data
    except Exception as e:
        if 'set enforce_detection' in str(e):
            return {'result': []}
        logging.error("Face representation error: %s", e, exc_info=True)
        return {'result': [], 'msg': str(e)}

@app.get("/", response_class=HTMLResponse)
async def top_info():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MT Photos AI Server</title>
    <style>p{text-align: center;}</style>
</head>
<body>
<p style="font-weight: 600;">MT Photos智能识别服务</p>
<p>服务状态： 运行中</p>
<p>使用方法： <a href="https://mtmt.tech/docs/advanced/ocr_api">https://mtmt.tech/docs/advanced/ocr_api</a></p>
</body>
</html>"""
    return html_content


@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {
        'result': 'pass',
        "title": "mt-photos-ai服务",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
        "detector_backend": detector_backend,
        "recognition_model": recognition_model,
        "facial_min_score": detection_thresh, # 推荐的人脸最低置信度阈值
        "facial_max_distance": 0.5, # 推荐的人脸差异值
    }


@app.post("/restart")
async def check_req(api_key: str = Depends(verify_header)):
    # 客户端可调用，触发重启进程来释放内存
    # restart_program()
    return {'result': 'pass'}

@app.post("/restart_v2")
async def check_req(api_key: str = Depends(verify_header)):
    # 预留触发服务重启接口-自动释放内存
    restart_program()
    return {'result': 'pass'}

@app.post("/ocr")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    image_bytes = await file.read()
    return await predict(_run_ocr_pipeline, image_bytes, executor=ocr_executor)


@app.post("/clip/img")
async def clip_process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    image_bytes = await file.read()
    return await predict(_clip_img_pipeline, image_bytes, executor=clip_executor)

@app.post("/clip/txt")
async def clip_process_txt(request:ClipTxtRequest, api_key: str = Depends(verify_header)):
    return await predict(_clip_txt_pipeline, request.text, executor=clip_executor)


@app.post("/represent")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    image_bytes = await file.read()
    return await predict(_run_represent_pipeline, image_bytes, file.content_type, file.filename, executor=face_executor)

def _represent(img):
    if face_model_pool is None:
        raise RuntimeError("Face model pool is not initialized.")
    face_analysis = face_model_pool.get()
    try:
        faces = face_analysis.get(img)
    finally:
        face_model_pool.put(face_analysis)

    results = []
    for face in faces:
        resp_obj = {}
        embedding = face.normed_embedding.astype(float)
        resp_obj["embedding"] = embedding.tolist()
        box = face.bbox
        resp_obj["facial_area"] = {"x" : int(box[0]), "y" : int(box[1]), "w" : int(box[2] - box[0]), "h" : int(box[3] - box[1])}
        resp_obj["face_confidence"] = face.det_score.astype(float)
        results.append(resp_obj)
    return results

def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)


def _create_warmup_image_bytes(width=640, height=640):
    """Create a dummy encoded image for warm-up."""
    dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
    success, buffer = cv2.imencode(".png", dummy_image)
    if not success:
        raise RuntimeError("Failed to encode warm-up image.")
    return buffer.tobytes()


def _warmup_models_sync():
    """Run one pass for every pipeline so workers are hot before serving traffic."""
    dummy_bytes = _create_warmup_image_bytes()
    _clip_img_pipeline(dummy_bytes)
    _run_ocr_pipeline(dummy_bytes)
    _run_represent_pipeline(dummy_bytes, "image/png", "warmup.png")
    _clip_txt_pipeline("warmup text")


async def warmup_models():
    """Execute synchronous warm-up in a background thread."""
    global models_warmed
    if models_warmed:
        return
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _warmup_models_sync)
    models_warmed = True
    print("[INFO] Model warm-up complete.")
