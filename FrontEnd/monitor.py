import streamlit as st
import pandas as pd
import requests
import time
import threading
from datetime import datetime, timedelta
from db_utils import execute_query, get_sleep_data
import plotly.graph_objects as go
from loadbox import LoadBox   # LoadBox 클래스 임포트   

# FastAPI 설정
IP = "http://192.168.0.14:8000/"
BE_START_URL = IP + "start"
BE_END_URL = IP + "end"

# 그래프 Y축 라벨 정의
POSE_LABELS = {0: "바로 누운 자세", 1: "옆으로 누워자기", 2: "팔든 자세", 3: "엎드린 자세", 4: "이외 자세"}
AUDIO_LABELS = {0: "일반/기타", 1: "코골이", 2: "이갈이"}


# # --- 콜백 함수 정의 (버튼 클릭 시 즉시 실행) ---
def start_monitoring_callback(new_uid, new_upw, new_ip):
    """시작 버튼 클릭 시 실행되는 콜백"""
    payload = {
        "login_id": st.session_state.user_id,
        "user_id": new_uid,
        "password": new_upw,
        "ip": new_ip
    }

    try:
        response = requests.post(BE_START_URL, json=payload)

        if response.json().get("code") == 200:
            st.toast("모니터링을 시작합니다.")

            st.session_state.is_analyzing = True
            st.session_state.start_time = time.time()
            st.session_state.start_time_dt = datetime.now() # 리포트(datetime) 조회를 위해 추가
        elif response.json().get("code") == 400:
            st.toast("이미 모니터링 중입니다!")
    except Exception as e:
        st.error(f"백엔드 연결 실패: {e}")

# loadbox simulation callback
# def start_monitoring_callback(new_uid, new_upw, new_ip):
#     user_id = st.session_state.get("user_id")
    
#     # LoadBox 인스턴스 세션 관리
#     if 'loadbox' not in st.session_state:
#         st.session_state.loadbox = LoadBox(user_id)
    
#     # 스레드 실행 (이미 실행 중이지 않을 때만)
#     if 'loadbox_thread' not in st.session_state or not st.session_state.loadbox_thread.is_alive():
#         st.session_state.loadbox_thread = threading.Thread(
#             target=st.session_state.loadbox.start_simulation, 
#             daemon=True
#         )
#         st.session_state.loadbox_thread.start()

#     st.session_state.is_analyzing = True
#     st.session_state.start_time = time.time() # 경과 시간(float) 계산용
#     st.session_state.start_time_dt = datetime.now() # 리포트(datetime) 조회를 위해 추가
#     st.toast("모니터링 시작: 가상 데이터를 30초마다 생성합니다.", icon="✅")

def stop_monitoring_callback():
    """종료 버튼 클릭 시 실행되는 콜백"""
    try:
        # FastAPI 종료 엔드포인트 호출
        # LoadBox 중단 (현재는 LoadBox 표시 안함)
        # if 'loadbox' in st.session_state:
        #     st.session_state.loadbox.stop_simulation()
        
        # 리포트 조회를 위해 종료 시간 기록 (TypeError 방지를 위해 _dt 통일 권장)
        response = requests.post(BE_END_URL)
        if response.json().get("code") == 200:
            st.toast("모니터링이 종료되었습니다.")

            st.session_state.last_report_data = response.json()
            
            st.session_state.end_time_dt = datetime.now() 
            st.session_state.is_analyzing = False
            st.session_state.page = "report" # 리포트 페이지로 이동
        elif response.json().get("code") == 500:
            st.toast("종료할 모니터링 스트림이 없습니다!")
     
    except Exception as e:
        st.error(f"종료 요청 실패: {e}")

# def stop_monitoring_callback():
#     # LoadBox 중단
#     if 'loadbox' in st.session_state:
#         st.session_state.loadbox.stop_simulation()
    
#     # 리포트 조회를 위해 종료 시간 기록 (TypeError 방지를 위해 _dt 통일 권장)
#     st.session_state.end_time_dt = datetime.now() 
#     st.session_state.is_analyzing = False
#     st.session_state.page = "report" # 리포트 페이지로 이동
#     st.toast("모니터링이 종료되었습니다.", icon="🛑")

@st.fragment(run_every=5.0)
def data_visualization_fragment(user_id):
    # 1. 데이터 가져오기
    df_pose, df_audio = get_sleep_data(user_id)
    
    # 2. 시간 기준 설정 (모두 타임존 없는 datetime으로 통일)
    now = datetime.now()
    start_time_dt = st.session_state.get("start_time_dt")
    window_limit = now - timedelta(minutes=10) # 30분을 10분으로 수정

    # 시작 시간이 없으면 현재를 기준으로 방어 코드 작성
    if start_time_dt is None:
        display_start_time = window_limit
    else:
        # 시작 시점부터 보되, 10분이 넘어가면 현재 기준 10분 전까지만 표시 (Sliding Window)
        display_start_time = max(start_time_dt, window_limit)

    # --- [핵심] 데이터 필터링 및 타입 변환 ---
    def process_and_filter(df, start_time):
        if df is not None and not df.empty:
            # st_dt 컬럼을 datetime 객체로 변환 (타입 불일치 해결)
            df['st_dt'] = pd.to_datetime(df['st_dt'])
            # 필터링 실행
            return df[df['st_dt'] >= start_time]
        return pd.DataFrame()

    df_pose = process_and_filter(df_pose, display_start_time)
    df_audio = process_and_filter(df_audio, display_start_time)
    # ---------------------------------------

    t1, t2 = st.tabs(["🛌 자세 분석 (최근 10분)", "🔊 소리 분석 (최근 10분)"])
    
    # --- 1) 자세(Pose) 계단식 그래프 ---
    with t1:
        if not df_pose.empty:
            fig_pose = go.Figure()
            fig_pose.add_trace(go.Scatter(
                x=df_pose['st_dt'], 
                y=df_pose['pose_class'],
                mode='lines+markers',
                line=dict(shape='hv', width=3, color='#00CC96'),
                marker=dict(size=6),
                name='수면 자세'
            ))
            
            fig_pose.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(
                    range=[display_start_time, now], 
                    type='date', # 축 타입을 명시적으로 지정
                    title="시간"
                ),
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(POSE_LABELS.keys()), 
                    ticktext=list(POSE_LABELS.values()) 
                ),
                hovermode="x unified"
            )
            st.plotly_chart(fig_pose, use_container_width=True)
        else:
            st.info(f"{display_start_time.strftime('%H:%M:%S')} 이후의 자세 데이터가 아직 없습니다.")

    # --- 2) 소리(Audio) 계단식 그래프 ---
    with t2:
        if not df_audio.empty:
            fig_audio = go.Figure()
            fig_audio.add_trace(go.Scatter(
                x=df_audio['st_dt'], 
                y=df_audio['audio_class'],
                mode='lines+markers',
                line=dict(shape='hv', width=3, color='#EF553B'),
                marker=dict(size=6),
                name='소리 이벤트'
            ))
            
            fig_audio.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(
                    range=[display_start_time, now], 
                    type='date', 
                    title="시간"
                ),
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(AUDIO_LABELS.keys()), 
                    ticktext=list(AUDIO_LABELS.values()) 
                ),
                hovermode="x unified"
            )
            st.plotly_chart(fig_audio, use_container_width=True)
        else:
            st.info(f"{display_start_time.strftime('%H:%M:%S')} 이후의 소리 데이터가 아직 없습니다.")

# --- 메인 윈도우 함수 ---
def monitoring_window():
    st.title("🌙 수면 모니터링 제어 센터")
    
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.warning("로그인이 필요합니다.")
        return

    # 세션 상태 초기화
    if 'is_analyzing' not in st.session_state:
        st.session_state.is_analyzing = False
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'start_time_dt' not in st.session_state:
        st.session_state.start_time_dt = None
    
    # --- [재배치 로직 시작] ---

    if not st.session_state.is_analyzing:
        # 1. 모니터링 시작 전: RTSP 설정창을 최상단에 표시
        st.subheader("📡 RTSP 설정")
        user_info = execute_query(
            "SELECT RTSP_ip_address, RTSP_user_id, RTSP_user_passwd FROM users WHERE user_id = %s",
            (user_id,), fetch_one=True
        )

        # 5.1 RTSP 정보 입력 섹션 (시작 전 화면용)
        with st.form(key=f"rtsp_config_form_{user_id}"):
            new_ip = st.text_input("카메라 IP 주소", value=user_info['RTSP_ip_address'] if user_info else "")
            new_uid = st.text_input("RTSP 사용자 ID", value=user_info['RTSP_user_id'] if user_info else "")
            new_upw = st.text_input("RTSP 비밀번호", value=user_info['RTSP_user_passwd'] if user_info else "", type="password")
            
            save_btn = st.form_submit_button("RTSP 정보 저장/업데이트")
            if save_btn:
                execute_query(
                    "UPDATE users SET RTSP_ip_address=%s, RTSP_user_id=%s, RTSP_user_passwd=%s WHERE user_id=%s",
                    (new_ip, new_uid, new_upw, user_id)
                )
                st.success("RTSP 정보가 업데이트되었습니다.")
                st.rerun()

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            # 5.2 모니터링 제어 섹션 - 시작 버튼 (설정창 아래에 배치)
            if st.button("▶️ 모니터링 시작", use_container_width=True, key="btn_start_monitoring"):
                start_monitoring_callback(new_uid, new_upw, new_ip)
                st.rerun()
        with col2:
            if st.button("📊 리포트 통계", use_container_width=True):
                st.session_state.page = 'summaryReport'
                st.rerun()          
    else:
        # 2. 모니터링 중: 실시간 분석 현황(그래프)을 최상단에 배치
        st.subheader("📊 실시간 분석 현황 (최근 10분)")
        # 정의된 외부 fragment 호출
        data_visualization_fragment(user_id)

        st.divider()

        # 3. 종료 버튼 및 경과 시간을 그래프 아래에 배치
        col_btn, col_timer = st.columns([1, 1])
        
        with col_btn:
            # 5.5 모니터링 종료 버튼
            st.button("⏹️ 모니터링 종료", 
                      use_container_width=True, 
                      on_click=stop_monitoring_callback,
                      key="btn_stop_monitoring")
        
        with col_timer:
            # 5.3 경과 시간 업데이트 (Fragment 사용)
            @st.fragment(run_every=1.0)
            def timer_fragment():
                if st.session_state.start_time:
                    elapsed_sec = int(time.time() - st.session_state.start_time)
                    elapsed_time = str(timedelta(seconds=elapsed_sec))
                    st.metric("⏳ 경과 시간", elapsed_time)
            
            timer_fragment()

    # --- [재배치 로직 끝] ---
    # 이 아래에 있던 기존의 5.1~5.5 중복 코드들은 위 if-else 문 내부로 통합되어 제거되었습니다.