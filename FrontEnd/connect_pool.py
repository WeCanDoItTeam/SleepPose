from dbutils.pooled_db import PooledDB
import mariadb
import streamlit as st
from contextlib import contextmanager
import threading

class DBPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._pool = cls.create_pool()
        return cls._instance

    # @staticmethod
    def create_pool():
        db_info = st.secrets["mariadb"]
        
        try:
            dbpool = PooledDB(
            creator=mariadb.connect,  # mariadb 커넥터
            maxconnections=20,  # 최대 연결수 ( 20유저)
            mincached=2,  # 최소 캐시 연결
            maxcached=5,  # 최대 캐시 연결
            maxshared=3,  # 공유 연결 최대
            blocking=True,  # 연결없으면 대기
            maxusage=0,  # 무제한 사용
            host=db_info["host"],
            port=db_info["port"],
            user=db_info["user"],
            password=db_info["password"],
            database=db_info["database"],
            # charset="utf8mb4",
            autocommit=True,
            # auth_plugin='mysql_native_password',
            setsession=[
                # 'SET NAMES utf8mb4',
                "SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED",  # 동시성
                "SET autocommit = 1",  # YOLO 30FPS 삽입
                "SET innodb_lock_wait_timeout = 10"  # 데드락 방지
            ] #자동세션 (연결될 때마다 실행)
            )
            return dbpool
        except mariadb.Error as e:
            print("❌ DB 연결 실패:", e)
            raise

    # @contextmanager
    def get_connection(self):
        conn = self._pool.connection()
        return conn
      