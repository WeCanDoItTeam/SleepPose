import sys
import os
import torch
import torchaudio.transforms as T
import numpy as np
import subprocess
import wave
from datetime import datetime, timedelta
from Inference_Server.inference.db_utils import get_db_connection

# --- 설정값 ---
SAMPLE_RATE = 16000
CHUNK_DURATION = 1  # 1초
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION * 2  # 16bit = 2bytes -> 32000 bytes
CONF_THRESHOLD = 0.85  # 시작 임계값 (60%)
SILENCE_TIMEOUT = 10  # 다른 이벤트나 소음이 지속되면 종료할 시간 (초)
RECORDING_DIR = "./recordings"  # 녹음 파일 저장 경로

# 디렉토리 생성
os.makedirs(RECORDING_DIR, exist_ok=True)

# --- 1. 전처리 및 모델 로드 (기존과 동일) ---
def rms_normalize(audio_chunk, target_rms=0.1):
    rms = np.sqrt(np.mean(audio_chunk**2))
    if rms < 1e-6:
        return audio_chunk
    gain = target_rms / rms
    return audio_chunk * gain

def preprocess_audio_chunk(audio_chunk):
    audio_chunk = rms_normalize(audio_chunk)
    if isinstance(audio_chunk, np.ndarray):
        audio_chunk = torch.from_numpy(audio_chunk).float()
    if len(audio_chunk.shape) == 1:
        audio_chunk = audio_chunk.unsqueeze(0)

    transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160,
        n_mels=64, f_min=125, f_max=7500, center=False
    )
    amp_to_db = T.AmplitudeToDB(top_db=80)

    spec = transform(audio_chunk)
    log_spec = amp_to_db(spec)
    log_spec = log_spec.permute(0, 2, 1)

    if log_spec.shape[1] > 96:
        log_spec = log_spec[:, :96, :]
    elif log_spec.shape[1] < 96:
        pad_h = 96 - log_spec.shape[1]
        log_spec = torch.nn.functional.pad(log_spec, (0, 0, 0, pad_h))

    return log_spec.unsqueeze(0)

def load_audio_model(model_path, device='cuda'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.join(base_dir, 'torch_yamnet')
    
    if repo_path not in sys.path:
        sys.path.append(repo_path)

    try:
        from torch_audioset.yamnet.model import yamnet
        model = yamnet(pretrained=False)
    except ImportError as e:
        print(f"❌ Import Failed: {e}")
        return None

    model.classifier = torch.nn.Linear(1024, 3)
    
    try:
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"   ✅ Audio Weights loaded: {model_path}")
        else:
            print(f"❌ Weights file not found: {model_path}")
            return None
    except Exception as e:
        print(f"❌ State Dict Load Error: {e}")
        return None
        
    model.to(device)
    model.eval()
    return model


# --- 2. 오디오 추론 엔진 (시간 계산 로직 수정) ---
class AudioInferenceEngine:
    def __init__(self, model, device, login_id):
        self.model = model
        self.device = device
        self.login_id = login_id
        
        self.is_recording = False
        self.start_event_class = None
        self.silence_counter = 0         

        self.audio_buffer = []           

        # 🎯 기준 시각 (영상 시작 시각과 동일해야 함)
        self.base_timestamp = datetime.now()

        # 🎯 실제 누적 오디오 시간 (초)
        self.processed_seconds = 0.0
        self.current_start_offset = 0.0

        self.session_timeline = []

        self.CLASS_NOISE = 0
        self.CLASS_SNORE = 1
        self.CLASS_BRUXISM = 2

    def process_chunk(self, audio_float, audio_bytes):
        # === 1. 실제 오디오 길이 계산 (핵심) ===
        chunk_sec = len(audio_bytes) / (SAMPLE_RATE * 2)

        # === 2. 추론 ===
        input_tensor = preprocess_audio_chunk(audio_float).to(self.device)
        with torch.no_grad():
            raw_output = self.model(input_tensor)
            outputs = raw_output[0] if isinstance(raw_output, (tuple, list)) else raw_output

            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            label = predicted.item()
            conf = confidence.item()

        # === 3. 상태 머신 ===
        if not self.is_recording:
            if label in [self.CLASS_SNORE, self.CLASS_BRUXISM] and conf >= CONF_THRESHOLD:
                print(f"🔊 [START] Audio Event {label} ({conf:.2f})")
                self._start_session(label, audio_bytes)

        else:
            self.audio_buffer.append(audio_bytes)

            if label == self.start_event_class:
                self.silence_counter = 0
            else:
                self.silence_counter += chunk_sec

            if self.silence_counter >= SILENCE_TIMEOUT:
                print(f"⏹ [END] Silence Timeout ({self.silence_counter:.2f}s)")
                self._end_session()

        # ✅ 실제 시간만큼 누적
        self.processed_seconds += chunk_sec

    def _start_session(self, label, first_chunk):
        self.is_recording = True
        self.start_event_class = label
        self.silence_counter = 0.0
        self.audio_buffer = [first_chunk]

        # 🎯 현재까지 흐른 실제 오디오 시간
        self.current_start_offset = self.processed_seconds

    def _end_session(self):
        trim_sec = min(self.silence_counter, self.processed_seconds)
        real_end_offset = self.processed_seconds - trim_sec

        # === 버퍼 trimming ===
        trim_chunks = int(trim_sec)  # 초 단위 기준
        if trim_chunks > 0:
            final_audio = self.audio_buffer[:-trim_chunks]
        else:
            final_audio = self.audio_buffer

        if not final_audio and self.audio_buffer:
            final_audio = self.audio_buffer[:1]

        # === 시간 계산 ===
        st_dt = self.base_timestamp + timedelta(seconds=self.current_start_offset)
        ed_dt = self.base_timestamp + timedelta(seconds=real_end_offset)

        # === 파일 저장 ===
        filename = f"{self.login_id}_{st_dt.strftime('%Y%m%d_%H%M%S')}_{self.start_event_class}.wav"
        filepath = os.path.join(RECORDING_DIR, filename)
        self._save_wav(filepath, final_audio)

        print(f"🕒 {st_dt} ~ {ed_dt}  ({len(final_audio)}s)")

        self.session_timeline.append({
            "class": self.start_event_class,
            "start": st_dt,
            "end": ed_dt,
            "path": filepath
        })

        # === reset ===
        self.is_recording = False
        self.audio_buffer = []
        self.start_event_class = None
        self.silence_counter = 0.0

    def _save_wav(self, filepath, buffer_list):
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(buffer_list))

    def force_close(self):
        if self.is_recording:
            self.silence_counter = 0.0
            self._end_session()
        return self.session_timeline



# --- 3. DB 일괄 저장 함수 ---
def save_audio_to_mariadb(login_id, session_data_list):
    """모아둔 오디오 세션 정보를 DB에 한 번에 저장"""
    if not session_data_list:
        print("⚠️ [Audio] 저장할 오디오 데이터가 없습니다.")
        return

    print(f"\n💾 [DB 저장] 유저 {login_id} 오디오 기록 {len(session_data_list)}건 저장 시작")

    conn = get_db_connection()
    if conn is None:
        print("❌ DB 연결 실패로 오디오 데이터 저장 중단")
        return

    try:
        with conn.cursor() as cur:
            # sleep_audio_log 컬럼에 파일 경로(data['path'])를 넣습니다.
            insert_sql = """
            INSERT INTO sleep_audio (user_id, audio_class, st_dt, ed_dt, sleep_audio_log, dt)
            VALUES (%s, %s, %s, %s, %s, %s)
            """

            rows = []
            now_dt = datetime.now()
            for data in session_data_list:
                rows.append((
                    login_id,
                    data['class'],
                    data['start'],
                    data['end'],
                    data['path'], # 파일 경로 문자열
                    now_dt
                ))

            cur.executemany(insert_sql, rows)
            conn.commit()
            print(f"✅ 오디오 DB 저장 완료 ({len(rows)}건)")

    except Exception as e:
        conn.rollback()
        print("❌ 오디오 DB 저장 실패:", e)
    finally:
        conn.close()


# --- 4. 메인 실행 함수 ---
def run_audio_inference(source, stop_flag, login_id, model_path="yamnet_finetuned_best.pth"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🎤 Audio Inference Started on {device} (Mode: Batch Save)")

    try:
        model = load_audio_model(model_path, device)
        if model is None: return
        engine = AudioInferenceEngine(model, device, login_id)
    except Exception as e:
        print(f"❌ Audio Init Error: {e}")
        return

    # FFmpeg 설정
    if os.path.isfile(source):
        cmd = ["ffmpeg", "-loglevel", "quiet", "-i", source, "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    else:
        cmd = ["ffmpeg", "-loglevel", "quiet", "-rtsp_transport", "tcp", "-i", source, "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    
    process = None
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        read_chunk_size = CHUNK_SIZE

        while not stop_flag():
            audio_bytes = process.stdout.read(read_chunk_size)
            if not audio_bytes: break
            
            if len(audio_bytes) < read_chunk_size:
                audio_bytes += b'\x00' * (read_chunk_size - len(audio_bytes))

            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            engine.process_chunk(audio_float, audio_bytes)
            
    except Exception as e:
        print(f"Audio Loop Error: {e}")
        
    finally:
        # 1. FFmpeg 종료
        if process: process.terminate()
        
        # 2. 진행 중이던 세션 강제 종료 및 데이터 확보
        final_timeline = engine.force_close()
        
        # 3. [NEW] DB 일괄 저장
        if final_timeline:
            save_audio_to_mariadb(login_id, final_timeline)
            
        print("🎤 Audio Inference Stopped Completely")

        