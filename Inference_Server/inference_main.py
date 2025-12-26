import os
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import threading
import time

# 1. 영상 추론 모듈 (기존 파일 유지)
# from Inference_Server.inference.inference_pose import run_ffmpeg_yolo
# 2. 오디오 추론 모듈 (새로 만든 파일)
from Inference_Server.inference.inference_audio import run_audio_inference

app = FastAPI()

FFMPEG_PATH = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"

class GlobalState:
    stream_start_time = None
    stream_end_time = None
    inference_running = False

state = GlobalState()

class UserData(BaseModel):
    login_id: str
    user_id: str
    password: str
    ip: str

# --- 비디오 스레드 함수 ---
# def video_thread_func(rtsp_url, login_id):
#     try:
#         run_ffmpeg_yolo(
#             rtsp_url=rtsp_url,
#             ffmpeg_path=FFMPEG_PATH,
#             stop_flag=lambda: not state.inference_running,
#             login_id=login_id
#         )
#     except Exception as e:
#         print(f"Video Thread Error: {e}")

# --- 오디오 스레드 함수 ---
def audio_thread_func(rtsp_url, login_id):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, "yamnet_finetuned_best.pth")
        print(f"🔍 Looking for weights at: {weights_path}")
        # 여기서 오디오 추론 메서드 실행 (오디오 로직 + FFmpeg Subprocess)
        run_audio_inference(
            source=rtsp_url,
            stop_flag=lambda: not state.inference_running,
            login_id=login_id,
            model_path=weights_path
        )
    except Exception as e:
        print(f"Audio Thread Error: {e}")

@app.post("/start")
async def start_inference(userData: UserData):
    if state.inference_running:
        return {"code": 400, "message": "Already running"}

    # rtsp_url = f"rtsp://{userData.user_id}:{userData.password}@{userData.ip}:554/stream2"
    # rtsp_url = "./data/oh_video/infer_Oh.mp4"
    rtsp_url = "./data/sample_audio/david_snoring.m4a"
        
    state.stream_start_time = datetime.now()
    state.inference_running = True

    # 1. 영상 스레드 시작
    # v_thread = threading.Thread(
    #     target=video_thread_func,
    #     args=(rtsp_url, userData.login_id),
    #     daemon=True
    # )
    # v_thread.start()

    # 2. 오디오 스레드 시작 (병렬 실행)
    a_thread = threading.Thread(
        target=audio_thread_func,
        args=(rtsp_url, userData.login_id),
        daemon=True
    )
    a_thread.start()

    return {"code": 200, "message": "Started"}

@app.post("/end")
async def end_inference():
    if not state.inference_running:
        return {"code": 500, "message": "Not running"}

    state.inference_running = False
    state.stream_end_time = datetime.now()

    return {"code": 200, "message": "Stopping"}

@app.get("/")
async def get_status():
    return {
        "inference_running": state.inference_running,
        "start_time": state.stream_start_time
    }