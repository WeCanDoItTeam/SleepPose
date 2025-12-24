from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import threading
from Inference_Server.inference.inference_pose import run_ffmpeg_yolo

# Fast API 선언
app = FastAPI()

# ffmpeg 경로 설정 (각자 PC마다 다름)
FFMPEG_PATH = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"

# 받는 User data 형식
class UserData(BaseModel):
    login_id: str
    user_id: str
    password: str
    ip: str

# 추론이 돌아가고 있는지 확인 및 설정용
class GlobalState:
    inference_running = False # 추론 돌아가고 있는지 여부
    inference_thread = None # 돌아가고 있는 추론 스레드
state = GlobalState()

# 추론 스트림 등록 메서드
def inference_loop(rtsp_url: str, login_id: str):
    # 자세 추론 스트림 실행
    run_ffmpeg_yolo(
        rtsp_url=rtsp_url,
        ffmpeg_path=FFMPEG_PATH,
        # 무명 메소드(lambda)라 inference_running 값이 변경되면 즉석에서 같이 변경됨
        # 스레드 종료 시 바로 반응하도록 callable 타입으로 매개변수 선언
        stop_flag=lambda: not state.inference_running,
        login_id= login_id
    )

    # 여기서 오디오 추론 메서드 실행


# 모니터링 시작 API
@app.post("/start")
async def start_inference(userData: UserData):
    if state.inference_running:
        return {"code": 400} # 이미 실행 중이면 400 에러 반환

    rtsp_url = f"rtsp://{userData.user_id}:{userData.password}@{userData.ip}:554/stream2"

    state.inference_running = True # 시작 bool 기록

    # 스레드 등록
    state.inference_thread = threading.Thread(
        target=inference_loop, # 여기에 스레드 등록되어 돌아감
        args=(rtsp_url, userData.login_id), # 매개변수 
        daemon=True
    )
    # 스레드 실행
    state.inference_thread.start() 

    return {"code": 200} # 정상 작동이면 200 반환

# 모니터링 종료 API
@app.post("/end")
async def end_inference():
    if not state.inference_running:
        return {"code": 500} # 종료 할 스트림이 없으면 500 에러 반환

    state.inference_running = False # 종료 bool 기록 (callable로 인한 반복문 자동 종료)

    return {"code": 200} # 정상 작동이면 200 반환

# 현재 상태 확인용 엔드포인트 (사용 안 함)
@app.get("/")
async def get_status():
    return {
        "inference_running": state.inference_running,
        "start_time": state.stream_start_time
    }
