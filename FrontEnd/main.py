# main.py
import streamlit as st
import db_utils
import account
import monitor
import report

# Streamlit 애플리케이션 설정
st.set_page_config(
    page_title="수면 모니터링 시스템",
    layout="centered"
)

# 세션 상태 초기화
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.page = 'login'

def login_form():
    """로그인 폼을 표시하고 인증을 처리합니다."""
    st.title("💡 수면 모니터링 시스템 - 로그인")

    # 1. 로그인 폼 (st.form_submit_button만 포함)
    with st.form("login_form"):
        user_id = st.text_input("사용자 ID")
        user_passwd = st.text_input("비밀번호", type="password")
        
        # 폼 제출 버튼
        submitted = st.form_submit_button("로그인")
        
        if submitted:
            # 2.1: 등록된 가입자인지 확인
            query = "SELECT user_id, user_passwd FROM users WHERE user_id = %s"
            user_data = db_utils.execute_query(query, (user_id,), fetch_one=True)
            
            if user_data and user_data['user_passwd'] == user_passwd:
                st.session_state.logged_in = True
                st.session_state.user_id = user_id
                st.session_state.page = 'monitor' # 등록된 가입자일 경우 "모니터링" 절차로
                st.success("로그인 성공!")
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 틀렸습니다.")

    # 2. 신규등록 버튼 (폼 외부로 이동)
    st.markdown("---")
    st.write("계정이 없으신가요?")
    if st.button("신규등록", key="new_account_btn"):
        st.session_state.page = 'account_creation'
        st.rerun()

# --- 페이지 라우팅 ---
if st.session_state.page == 'login':
    db_utils.initialize_db() # DB 초기화 체크
    login_form()
elif st.session_state.page == 'account_creation':
    account.account_creation_window()
elif st.session_state.page == 'monitor':
    if st.session_state.logged_in:
        monitor.monitoring_window()
    else:
        st.session_state.page = 'login'
        st.warning("로그인이 필요합니다.")
        st.rerun()
elif st.session_state.page == 'report':
    if st.session_state.logged_in:
        report.report_window()
    else:
        st.session_state.page = 'login'
        st.warning("로그인이 필요합니다.")
        st.rerun()

# 로그아웃 버튼 (로그인 시에만 사이드바에 표시)
if st.session_state.logged_in:
    def logout():
        # 모니터링 세션 데이터 초기화
        keys_to_clear = ['monitoring_running', 'start_time', 'end_time', 'sleep_data', 'rtsp_tick', 
                         'pose_count', 'pose_duration', 'audio_count', 'audio_duration',
                         'last_pose_data', 'last_audio_data']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
                
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.page = 'login'
        
    st.sidebar.button("로그아웃", on_click=logout)