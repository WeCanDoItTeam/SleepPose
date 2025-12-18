import time
import random
from datetime import datetime
import mariadb
from db_utils import get_db_connection, execute_query

class LoadBox:
    def __init__(self, user_id):
        self.is_running = False
        self.user_id = user_id
        self.last_pose_id = None
        self.last_audio_id = None

    def start_simulation(self):
        """30초 주기로 랜덤 데이터를 생성하고 DB에 기록합니다."""
        self.is_running = True
        print(f"--- [LoadBox] {self.user_id} 시뮬레이션 시작 ---")
        
        try:
            while self.is_running:
                now = datetime.now()
                today = now.date()

                # 1. 이전 기록이 있다면 ed_dt(종료시간)를 현재 시간으로 업데이트
                if self.last_pose_id:
                    execute_query("UPDATE sleep_pose SET ed_dt = %s WHERE id = %s", (now, self.last_pose_id))
                if self.last_audio_id:
                    execute_query("UPDATE sleep_audio SET ed_dt = %s WHERE id = %s", (now, self.last_audio_id))

                # 2. 새로운 랜덤 데이터 생성 (스키마에 정의된 클래스 범위 준수)
                pose_val = random.randint(0, 4)   # 0~4: 바로누움, 옆으로, 팔든, 엎드린, 이외
                audio_val = random.randint(0, 2)  # 0~2: Others, 코골이, 이갈이

                # 3. 새로운 레코드 삽입 및 ID 보관 (다음 루프에서 ed_dt 업데이트를 위함)
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Pose Insert
                cursor.execute(
                    "INSERT INTO sleep_pose (dt, pose_class, st_dt, user_id) VALUES (%s, %s, %s, %s)",
                    (today, pose_val, now, self.user_id)
                )
                self.last_pose_id = cursor.lastrowid
                
                # Audio Insert
                cursor.execute(
                    "INSERT INTO sleep_audio (dt, audio_class, st_dt, user_id) VALUES (%s, %s, %s, %s)",
                    (today, audio_val, now, self.user_id)
                )
                self.last_audio_id = cursor.lastrowid
                
                conn.commit()
                cursor.close()
                conn.close()

                print(f"[LoadBox] Update: Pose({pose_val}), Audio({audio_val}) at {now.strftime('%H:%M:%S')}")
                
                # 30초 대기 (1초씩 끊어서 체크해야 '종료' 클릭 시 즉각 반응함)
                for _ in range(30):
                    if not self.is_running: break
                    time.sleep(1)
                    
        except Exception as e:
            print(f"LoadBox 실행 중 오류: {e}")
        finally:
            self.stop_simulation()

    def stop_simulation(self):
        """시뮬레이션을 멈추고 마지막 데이터의 종료 시간을 기록합니다."""
        if self.is_running: # 실행 중이었을 때만 종료 처리
            now = datetime.now()
            if self.last_pose_id:
                execute_query("UPDATE sleep_pose SET ed_dt = %s WHERE id = %s", (now, self.last_pose_id))
            if self.last_audio_id:
                execute_query("UPDATE sleep_audio SET ed_dt = %s WHERE id = %s", (now, self.last_audio_id))
            self.is_running = False
            print(f"--- [LoadBox] {self.user_id} 시뮬레이션 종료 ---")