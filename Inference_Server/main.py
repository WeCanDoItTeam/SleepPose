from datetime import datetime
import threading
from inference.inference_pose import run_ffmpeg_yolo

FFMPEG_PATH = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"

# 전역 상태
stream_start_time = None
stream_end_time = None
inference_running = False
inference_thread = None

# 추론 호출
def inference_loop(rtsp_url: str):
    global inference_running

    run_ffmpeg_yolo(
        rtsp_url=rtsp_url,
        ffmpeg_path=FFMPEG_PATH,
        stop_flag=lambda: not inference_running
    )

    # 여기에 음성 추론도 비슷하게 호출해서 하면 됨


# 스트림 시작
def start_inference(user_id: str, password: str, ip: str):
    global stream_start_time, inference_running, inference_thread

    if inference_running:
        return 400 # 이미 실행되고 있는 스트림이 있음 (예시 리턴입니다)

    rtsp_url = f"rtsp://{user_id}:{password}@{ip}:554/stream2"

    stream_start_time = datetime.now()
    inference_running = True

    inference_thread = threading.Thread(
        target=inference_loop,
        args=(rtsp_url,),
        daemon=True
    )
    inference_thread.start()

    return 200 #정상 스트림 연결 (예시 리턴입니다)


# 스트림 종료
def end_inference():
    global stream_end_time, inference_running

    if not inference_running:
        return 500 # 종료 할 스트림이 없음 (예시 리턴입니다)

    inference_running = False
    stream_end_time = datetime.now()

    # UI측에서 해당 시작시간-종료시간으로 DB 조회
    # 도커 서버 배포를 기준으로 작성된 것이나, 현재는 로컬서버로 우선 개발됩니다.
    # 로컬서버이므로 이 부분에서 바로 DB를 조회해 전체 데이터를 반환할 수도 있습니다.
    return {
        # 아래 타임스탬프로 DB의 포즈와 오디오 테이블 각각 조회
        "start_time": stream_start_time.isoformat(),
        "end_time": stream_end_time.isoformat()
    }
