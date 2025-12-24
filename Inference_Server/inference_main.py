from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import threading
from Inference_Server.inference.inference_pose import run_ffmpeg_yolo

app = FastAPI()

# 설정 및 전역 상태
FFMPEG_PATH = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"

class GlobalState:
    stream_start_time = None # 최초 시작 시간
    stream_end_time = None # 최종 종료 시간
    inference_running = False # 추론 돌아가고 있는지 여부
    inference_thread = None # 돌아가고 있는 추론 스레드

state = GlobalState()

def inference_loop(rtsp_url: str, login_id: str):
    # 전역 상태를 참조하여 루프 실행
    # try:
    run_ffmpeg_yolo(
        rtsp_url=rtsp_url,
        ffmpeg_path=FFMPEG_PATH,
        stop_flag=lambda: not state.inference_running,
        login_id= login_id
    )

        # 여기서 오디오 추론 메서드 실행

    # finally:
    #     state.inference_running = False
    #     print("Inference loop terminated.")

class UserData(BaseModel):
    login_id: str
    user_id: str
    password: str
    ip: str

@app.post("/start")
async def start_inference(userData: UserData):
    if state.inference_running:
        return {"code": 400} # 이미 실행 중이면 400 에러 반환

    rtsp_url = f"rtsp://{userData.user_id}:{userData.password}@{userData.ip}:554/stream2"

    state.stream_start_time = datetime.now()
    state.inference_running = True

    state.inference_thread = threading.Thread(
        target=inference_loop,
        args=(rtsp_url,userData.login_id),
        daemon=True
    )
    state.inference_thread.start()

    return {"code": 200}

@app.post("/end")
async def end_inference():
    if not state.inference_running:
        return {"code": 500} # 종료 할 스트림이 없음 (예시 리턴입니다)

    state.inference_running = False
    state.stream_end_time = datetime.now()

    # 스레드가 종료될 때까지 기다리지 않고 즉시 응답 (데몬 스레드이므로 플래그에 의해 종료됨)
    return {"code": 200}

# 현재 상태 확인용 엔드포인트 (스트림릿에서 유용함)
@app.get("/")
async def get_status():
    return {
        "inference_running": state.inference_running,
        "start_time": state.stream_start_time
    }
