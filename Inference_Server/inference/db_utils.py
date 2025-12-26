# db_utils.py
import mariadb
import pandas as pd
import Inference_Server.inference.connect_pool as dbPool

def get_db_connection():
    """데이터베이스 연결 객체를 반환합니다. (매번 새로 연결하여 안정성 확보)"""
    try:
        pool = dbPool.DBPool()               # 싱글톤 풀 인스턴스
        return pool.get_connection()  # 여기서 "진짜 커넥션"을 리턴
     
    except mariadb.Error as e:
        print(f"MariaDB 연결 오류: {e}")
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
        print(f"쿼리 실행 오류: {e}")
        return None
    finally:
        if conn:
            conn.close()
