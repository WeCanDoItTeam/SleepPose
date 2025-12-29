import streamlit as st
import db_utils
from datetime import datetime, timedelta
import pandas as pd
from db_utils import get_db_connection, get_sleep_data
from report_llm import get_llm

# 포즈/오디오 클래스 정의 재사용
POSE_CLASSES = {
    0: "바로 누운 자세", 
    1: "옆으로 누운 자세", 
    2: "팔든 자세", 
    3: "엎드린 자세", 
    4: "이외 자세"
}

AUDIO_CLASSES = {
    0: "일반/기타", 
    1: "코골이", 
    2: "이갈이"
}

def load_session_data(user_id, start_time, end_time):
    """주어진 시간 범위 내의 포즈/오디오 데이터를 DB에서 불러옵니다."""
    pose_query = """
        SELECT pose_class, st_dt, ed_dt 
        FROM sleep_pose 
        WHERE user_id = %s AND st_dt >= %s AND ed_dt <= %s
    """
    pose_data = db_utils.execute_query(pose_query, (user_id, start_time, end_time), fetch_all=True)

    audio_query = """
        SELECT audio_class, st_dt, ed_dt 
        FROM sleep_audio 
        WHERE user_id = %s AND st_dt >= %s AND ed_dt <= %s
    """
    audio_data = db_utils.execute_query(audio_query, (user_id, start_time, end_time), fetch_all=True)
    
    return pose_data, audio_data


def calculate_report_stats(data, class_map):
    """데이터 목록에서 횟수와 총 지속시간을 계산합니다."""
    stats = {k: {'count': 0, 'duration': timedelta(0)} for k in class_map}
    total_duration = timedelta(0)
    
    if not data:
        return pd.DataFrame(), timedelta(0) 

    # 데이터 키 확인
    class_key = 'pose_class' if 'pose_class' in data[0] else 'audio_class'

    for row in data:
        class_id = row[class_key]
        st_dt = row['st_dt']
        ed_dt = row['ed_dt']
        
        if st_dt and ed_dt:
            duration = ed_dt - st_dt
            if class_id in stats: # 정의된 클래스 내에 있을 때만 계산
                stats[class_id]['count'] += 1
                stats[class_id]['duration'] += duration
                total_duration += duration

    report_data = []
    for class_id, class_name in class_map.items():
        count = stats[class_id]['count']
        duration_obj = stats[class_id]['duration']
        duration_hms = str(duration_obj).split('.')[0]
        
        if total_duration.total_seconds() > 0:
            percentage = (duration_obj.total_seconds() / total_duration.total_seconds()) * 100
        else:
            percentage = 0
            
        report_data.append({
            '분류': class_name,
            '관찰 횟수': count,
            '총 지속 시간': duration_hms,
            '비중 (%)': f"{percentage:.1f}%"
        })
        
    return pd.DataFrame(report_data), total_duration

def report_window():
    user_id = st.session_state.user_id
    st.title(f"📊 {user_id}님의 수면 분석 리포트")
    
    # 1. 시간 정보 가져오기
    start_time = st.session_state.get('start_time_dt')
    end_time = st.session_state.get('end_time_dt')

    # 시간 정보가 없을 경우 처리
    if not start_time or not end_time:
        st.warning("최근 모니터링 세션 정보를 찾을 수 없어 최근 24시간 데이터를 표시합니다.")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24) 

    # --- [수정 포인트] 요약 카드 배치 ---
    # 시작 시간, 종료 시간, 총 모니터링 시간을 상단에 배치
    total_duration = end_time - start_time
    duration_hms = str(total_duration).split('.')[0]

    st.subheader("⏱️ 모니터링 요약")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ 시작 시각", start_time.strftime("%H:%M:%S"))
    with col2:
        st.metric("🛑 종료 시각", end_time.strftime("%H:%M:%S"))
    with col3:
        st.metric("⏳ 총 분석 시간", duration_hms)
    
    st.info(f"분석 일자: {start_time.strftime('%Y년 %m월 %d일')}")
    st.markdown("---")

    # 2. 데이터 로드
    pose_data, audio_data = load_session_data(user_id, start_time, end_time)

    # 3. Pose별 리포트
    st.subheader("🛌 자세 (Pose) 분석 결과")
    if pose_data:
        pose_report_df, _ = calculate_report_stats(pose_data, POSE_CLASSES)
        st.table(pose_report_df) # dataframe 보다 표가 리포트에 적합할 수 있음
    else:
        st.info("해당 기간 동안 분석된 자세 데이터가 없습니다.")

    st.markdown("---")

    # 4. Audio별 리포트
    st.subheader("🔊 소리 (Audio) 분석 결과")
    if audio_data:
        audio_report_df, _ = calculate_report_stats(audio_data, AUDIO_CLASSES)
        st.table(audio_report_df)
    else:
        st.info("해당 기간 동안 분석된 오디오 데이터가 없습니다.")
        
    st.markdown("---")

    # 하단 버튼 배치
    col_home, col_rerun = st.columns(2)
    with col_home:
        if st.button("🏠 메인 화면으로", use_container_width=True):
            st.session_state.page = 'login'
            st.rerun()
    with col_rerun:
        if st.button("🔄 리포트 새로고침", use_container_width=True):
            st.rerun()

    st.subheader("🤖 AI 수면 코칭 요약")

    with st.spinner("AI가 수면 데이터를 분석 중입니다..."):
        llm_response = get_llm((pose_data, audio_data))

    st.markdown(llm_response)


