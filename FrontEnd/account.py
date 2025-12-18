# account.py
import streamlit as st
import db_utils
import re

# 요구사항 4.1에 따른 정규 표현식 정의
# user_id: 영문 대소문자 & 숫자 (max 20)
USER_ID_REGEX = re.compile(r'^[a-zA-Z0-9]{1,20}$') 
# user_passwd: 영문 대소문자, 숫자, $, %, * (max 20)
USER_PASSWD_REGEX = re.compile(r'^[a-zA-Z0-9$%*]{1,20}$') 
# RTSP_ip_address: 숫자, . (max 15) -> IP 주소 형식 검증으로 대체
RTSP_IP_REGEX = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
# RTSP_user_id: 영문 대소문자 & 숫자 (min 6, max 32)
RTSP_USER_ID_REGEX = re.compile(r'^[a-zA-Z0-9]{6,32}$') 
# RTSP_user_passwd: 영문 대소문자, 숫자, !@#$%^&* (min 6, max 32)
RTSP_PASSWD_REGEX = re.compile(r'^[a-zA-Z0-9!@#$%^&*]{6,32}$')

def validate_input(field_name, value, regex, min_len=1, max_len=20):
    """입력 값의 유효성을 검사합니다."""
    if not (min_len <= len(value) <= max_len):
        return f"{field_name}는 {min_len}자에서 {max_len}자 사이여야 합니다."
    if not regex.match(value):
        return f"{field_name}의 허용 문자열이 올바르지 않습니다."
    return None

def account_creation_window():
    st.title("👤 신규 Account 생성")

    with st.form("new_account_form"):
        # 4.2: user_id, user_passwd 입력
        st.subheader("계정 정보")
        new_user_id = st.text_input("새 사용자 ID")
        new_user_passwd = st.text_input("비밀번호", type="password")
        confirm_passwd = st.text_input("비밀번호 확인", type="password")
        
        # 4.3: RTSP 정보 입력
        st.subheader("RTSP 카메라 정보 (선택 사항)")
        rtsp_ip = st.text_input("RTSP IP 주소 (예: 192.168.1.1)")
        rtsp_uid = st.text_input("RTSP 사용자 ID")
        rtsp_upw = st.text_input("RTSP 비밀번호", type="password")

        submitted = st.form_submit_button("가입 및 모니터링 시작")

        if submitted:
            errors = []

            # 비밀번호 확인
            if new_user_passwd != confirm_passwd:
                errors.append("비밀번호와 비밀번호 확인이 일치하지 않습니다.")

            # 유효성 검사
            errors.append(validate_input("사용자 ID", new_user_id, USER_ID_REGEX, 1, 20))
            errors.append(validate_input("비밀번호", new_user_passwd, USER_PASSWD_REGEX, 1, 20))
            errors.append(validate_input("RTSP IP", rtsp_ip, RTSP_IP_REGEX, 7, 15))
            errors.append(validate_input("RTSP ID", rtsp_uid, RTSP_USER_ID_REGEX, 6, 32))
            errors.append(validate_input("RTSP PW", rtsp_upw, RTSP_PASSWD_REGEX, 6, 32))

            errors = [e for e in errors if e is not None]
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # DB 중복 체크
                check_query = "SELECT user_id FROM users WHERE user_id = %s"
                if db_utils.execute_query(check_query, (new_user_id,), fetch_one=True):
                    st.error(f"'{new_user_id}'는 이미 존재하는 ID입니다.")
                    return

                # DB 저장
                try:
                    insert_query = """
                        INSERT INTO users 
                        (user_id, user_passwd, RTSP_ip_address, RTSP_user_id, RTSP_user_passwd) 
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    db_utils.execute_query(
                        insert_query, 
                        (new_user_id, new_user_passwd, rtsp_ip, rtsp_uid, rtsp_upw)
                    )
                    
                    st.success("신규 계정 등록 및 정보 저장 완료!")
                    
                    # 4.4: "모니터링" 절차로
                    st.session_state.logged_in = True
                    st.session_state.user_id = new_user_id
                    st.session_state.page = 'monitor'
                    st.rerun()

                except Exception as e:
                    st.error(f"DB 저장 중 오류 발생: {e}")