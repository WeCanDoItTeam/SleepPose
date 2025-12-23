# db_utils.py
import mariadb
import streamlit as st
import pandas as pd
import connect_pool as dbPool

def get_db_connection():
    """데이터베이스 연결 객체를 반환합니다. (매번 새로 연결하여 안정성 확보)"""
    try:
        pool = dbPool.DBPool()               # 싱글톤 풀 인스턴스
        return pool.get_connection()  # 여기서 "진짜 커넥션"을 리턴
     
    except mariadb.Error as e:
        st.error(f"MariaDB 연결 오류: {e}")
        return None

def execute_query(query, params=None, fetch_one=False, fetch_all=False):
    """일반적인 SELECT, UPDATE, INSERT 쿼리를 실행하는 통합 함수"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        # dictionary=True 옵션을 주면 결과값을 {'column': value} 형태로 받아올 수 있어 편리합니다.
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params)
        if fetch_one:
            result = cursor.fetchone()
        elif fetch_all:
            result = cursor.fetchall()
        else:
            conn.commit()
            result = True
            
        cursor.close()
        conn.close()
        return result
    except mariadb.Error as e:
        st.error(f"쿼리 실행 오류: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_sleep_data(user_id, limit=50):
    """
    monitor.py에서 차트를 그리기 위해 사용.
    최신 데이터를 판다스 데이터프레임으로 반환합니다.
    """
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame(), pd.DataFrame()

    try:
        # 1. 자세 데이터 가져오기
        pose_query = """
            SELECT st_dt, pose_class 
            FROM sleep_pose 
            WHERE user_id = %s 
            ORDER BY st_dt DESC LIMIT %s
        """
        # mariadb connector와 pandas 연동
        df_pose = pd.read_sql(pose_query, conn, params=(user_id, limit))
        
        # 2. 소리 데이터 가져오기
        audio_query = """
            SELECT st_dt, audio_class 
            FROM sleep_audio 
            WHERE user_id = %s 
            ORDER BY st_dt DESC LIMIT %s
        """
        df_audio = pd.read_sql(audio_query, conn, params=(user_id, limit))

        # 차트는 시간 순서(과거->현재)대로 그려야 하므로 재정렬
        if not df_pose.empty:
            df_pose = df_pose.sort_values('st_dt')
        if not df_audio.empty:
            df_audio = df_audio.sort_values('st_dt')

        return df_pose, df_audio

    except Exception as e:
        st.error(f"데이터 프레임 로드 실패: {e}")
        return pd.DataFrame(), pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_sleep_day(user_id,st_dt,ed_dt):
    """
    monitor.py에서 차트를 그리기 위해 사용.
    최신 데이터를 판다스 데이터프레임으로 반환합니다.
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor(dictionary=True)
        # 1. 자세 데이터 가져오기
        pose_query = """
            SELECT st_dt, pose_class, ed_dt
            FROM sleep_pose2
            WHERE user_id = %s 
            AND st_dt BETWEEN %s AND %s
        """
        st_dt = st_dt + " 13:00:00"
        ed_dt = ed_dt + " 13:00:00"
        cursor.execute(pose_query, (user_id, st_dt, ed_dt))
        data = cursor.fetchall()
        return data
    except Exception as e:
        st.error(f"데이터 프레임 로드 실패: {e}")
        return pd.DataFrame(), pd.DataFrame()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()            

def get_sleep_month(user_id,st_dt,ed_dt):
    """
    monitor.py에서 차트를 그리기 위해 사용.
    최신 데이터를 판다스 데이터프레임으로 반환합니다.
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor(dictionary=True)
        # 1. 자세 데이터 가져오기
        pose_query = """
            SELECT pose_class,c.code_nm as pose_nm,
                DATE_FORMAT(st_dt, '%H') AS hour_slot,
                SUM(TIMESTAMPDIFF(SECOND, st_dt, ed_dt))/60 AS minutes
            FROM sleep_pose2 t
            left outer join comm_code c on t.pose_class = c.code_id and c.code_cd='pose'
            WHERE st_dt BETWEEN %s AND %s
            AND user_id = %s
            GROUP BY hour_slot, pose_class, code_nm
        """
        st_dt = st_dt + " 13:00:00"
        ed_dt = ed_dt + " 13:00:00"
        cursor.execute(pose_query, (st_dt, ed_dt,user_id))
        data = cursor.fetchall()
        return data
    except Exception as e:
        st.error(f"데이터 프레임 로드 실패: {e}")
        return pd.DataFrame(), pd.DataFrame()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()             
            
def initialize_db():
    """앱 시작 시 테이블 구조를 자동으로 생성/확인합니다."""
    # 사용자의 코드와 동일하게 유지하되, dictionary=True 관련 이슈 방지를 위해 일반 cursor 사용
    conn = get_db_connection()
    if not conn: return
    
    try:
        cursor = conn.cursor()
        # users 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(20) PRIMARY KEY,
                user_passwd VARCHAR(20) NOT NULL,
                RTSP_ip_address VARCHAR(15),
                RTSP_user_id VARCHAR(32),
                RTSP_user_passwd VARCHAR(32)
            );
        """)
        # sleep_pose 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sleep_pose (
                id INT AUTO_INCREMENT PRIMARY KEY,
                dt DATE NOT NULL,
                pose_class TINYINT NOT NULL,
                st_dt TIMESTAMP NOT NULL,
                ed_dt TIMESTAMP,
                user_id VARCHAR(20),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
        """)
        # sleep_audio 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sleep_audio (
                id INT AUTO_INCREMENT PRIMARY KEY,
                dt DATE NOT NULL,
                audio_class TINYINT NOT NULL,
                st_dt TIMESTAMP NOT NULL,
                ed_dt TIMESTAMP,
                user_id VARCHAR(20),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
        """)
        conn.commit()
        cursor.close()
        
    except mariadb.Error as e:
        st.error(f"DB 초기화 실패: {e}")
    finally:
        if conn:
            conn.close()