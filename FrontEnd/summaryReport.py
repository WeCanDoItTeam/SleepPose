import streamlit as st
import pandas as pd
from db_utils import get_sleep_day, get_sleep_month
import plotly.graph_objects as go
import plotly.express as px
import calendar
from datetime import datetime, timedelta,date
def startEndDate(yyyymm):
    year = int(yyyymm[:4])
    month = int(yyyymm[4:])
    start_date = date(year, month, 1)  # 2025-03-01
    last_day = calendar.monthrange(year, month)[1]
    end_date = date(year, month, last_day)
    return (start_date ,end_date)
def addDate(yyyymmdd):
    d = datetime.strptime(yyyymmdd, "%Y-%m-%d").date()
    next_day = d + timedelta(days=1)
    return str(next_day)

def ganttchart(user_id, st_dt):
    ed_dt = addDate(st_dt)
    # 1. 데이터 가져오기
    pose_data = get_sleep_day(user_id,st_dt,ed_dt)
    pose_df = pd.DataFrame(pose_data)
    if pose_df.empty:
        st.markdown("## 데이터가 없습니다")
        return
    st.markdown("### 그래프 필터링 옵션")
    pose_df['st_dt'] = pd.to_datetime(pose_df['st_dt'])
    pose_df['ed_dt'] = pd.to_datetime(pose_df['ed_dt'])
    pose_df['pose_class'] = pose_df['pose_class'].astype(str)
    
    min_t = pose_df['st_dt'].min()
    max_t = pose_df['ed_dt'].max()
    
    # 1) Streamlit에서 깔끔한 범위 선택 UI
    start_t, end_t = st.slider(
        '표시할 시간 범위',
        min_value=min_t.to_pydatetime(),
        max_value=max_t.to_pydatetime(),
        value=(min_t.to_pydatetime(), max_t.to_pydatetime()),
        format="HH:mm:ss",
        key=f"time_range_{user_id}",
    )
    st.markdown("---")
    # 2) 선택된 범위로 데이터 필터
    mask = (pose_df['st_dt'] >= start_t) & (pose_df['st_dt'] <= end_t)
    view_df = pose_df[mask].copy()
    
    fig = px.timeline(view_df , x_start='st_dt', x_end='ed_dt', 
                    y='pose_class',
                    color='pose_class',
                    color_discrete_map={
                        '0': '#1f77b4',
                        '1': '#ff7f0e',
                        '2': '#2ca02c',
                        '3': "#9432d6",
                        '4': "#b0cf3f",
                    },
                    title="SleepPoseP 시분초 Gantt 차트")
    fig.update_xaxes(type='date'
                     , tickformat='%H시'
                     , dtick=3600*1000
                     ) # 5분 간격 (300*1000ms) 60*60=3600 )  # x축은 타입/포맷만
    fig.update_layout(
        xaxis_title='시간대(시)',  # 12, 13, 14 ...
        yaxis_title='',
        showlegend=False
    )
    # fig.update_yaxes(autorange='reversed')                # y축을 뒤집기
    # fig.update_layout(
    # xaxis=dict(
    #     type='date',
    #     rangeslider=dict(visible=True),  # 그래프 아래 슬라이더
    #     rangeselector=dict(             # 빠른 선택 버튼 (선택 사항)
    #         buttons=list([
    #             dict(count=5, label="5m", step="minute", stepmode="backward"),
    #             dict(count=30, label="30m", step="minute", stepmode="backward"),
    #             dict(step="all", label="All")
    #         ])
    #         )
    #     )
    # )
    st.plotly_chart(fig, use_container_width=True)  

def heatmapChart(user_id, st_dt):
    #1달간 시간대별 자세 소요시간 집계
    (start_date ,end_date) = startEndDate(st_dt)
    pose_data = get_sleep_month(user_id,str(start_date),str(end_date))
    pose_df = pd.DataFrame(pose_data)

    if pose_df.empty:
        st.markdown("## 데이터가 없습니다")
        return

    pose_df['hour_slot'] = pose_df['hour_slot'].astype(int)#시간대별
    pose_df['pose_class'] = pose_df['pose_class'].astype(str)
    pose_df['minutes'] = pose_df['minutes'].astype(float)#소요시간(분)
    pose_df['pose_nm'] = pose_df['pose_nm'].astype(str)
    
    fig = px.density_heatmap(
        pose_df,
        x='hour_slot',
        y='pose_nm',
        z='minutes',
        color_continuous_scale='Viridis',
        title='한달간 시간대별 자세 소요시간 Heatmap(분)'
    )
    
    fig.update_layout(
        xaxis_title='시간대(시)',      # 12, 13, 14 ...
        yaxis_title='포즈 클래스',     # 0,1,2,3,4
        coloraxis_colorbar_title='시간합(분)'  # 색바 라벨
    )
    st.plotly_chart(fig, use_container_width=True)  
def pieChart(user_id,  st_dt):
    (start_date, end_date) = startEndDate(st_dt)
    pose_data = get_sleep_month(user_id, str(start_date), str(end_date))
    pose_df = pd.DataFrame(pose_data)

    if pose_df.empty:
        st.markdown("## 데이터가 없습니다")
        return

    pose_df['hour_slot'] = pose_df['hour_slot'].astype(int)#시간대별
    pose_df['pose_class'] = pose_df['pose_class'].astype(str)
    pose_df['minutes'] = pose_df['minutes'].astype(float)#소요시간(분)
    pose_df['pose_nm'] = pose_df['pose_nm'].astype(str)
    

    labels =pose_df['pose_nm'] .tolist()
    values = pose_df['minutes'] .tolist()

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                                insidetextorientation='radial'
                                )])
    st.plotly_chart(fig, use_container_width=True)  
def barChart(user_id,  st_dt):
    (start_date, end_date) = startEndDate(st_dt)
    pose_data = get_sleep_month(user_id, str(start_date), str(end_date))
    pose_df = pd.DataFrame(pose_data)

    if pose_df.empty:
        st.markdown("## 데이터가 없습니다")
        return

    pose_df['hour_slot'] = pose_df['hour_slot'].astype(int)#시간대별
    pose_df['pose_class'] = pose_df['pose_class'].astype(str)
    pose_df['minutes'] = pose_df['minutes'].astype(float)#소요시간(분)
    pose_df['pose_nm'] = pose_df['pose_nm'].astype(str)
    

    labels =pose_df['pose_nm'] .tolist()
    values = pose_df['minutes'] .tolist()

    long_df = px.data.medals_long()
    fig = px.bar(pose_df, 
             x="hour_slot", 
             y="minutes", 
             color="pose_nm",
             barmode='stack',  # 병렬 막대
             title="시간대별 포즈 소요시간(분)",
             text="minutes",
             color_discrete_map={
                 '바로 누운 자세': '#1f77b4',
                 '옆으로 누워자기': '#ff7f0e',
                 '팔든 자세': '#2ca02c',
                 '엎드린 자세': "#9432d6",
                 '기타': "#b0cf3f"
             }
    )
    fig.update_layout(
        xaxis_title='시간대(시)',      # 12, 13, 14 ...
        yaxis_title='포즈 클래스'    # 0,1,2,3,4
        # coloraxis_colorbar_title='시간합(분)'  # 색바 라벨
    )
    # wide_df = px.data.medals_wide()
    # fig = px.bar(pose_df, x=values, y=labels, title="Wide-Form Input")
    st.plotly_chart(fig, use_container_width=True)  

def report_window():
    user_id = st.session_state.user_id
    reportFlag = st.session_state.reportFlag #default:D
    selected_chart = None
    selected_date = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🕒일간 리포트", use_container_width=True):
            st.session_state.page = 'summaryReport'
            st.session_state.reportFlag = 'D'
            st.rerun()
    with col2:
        if st.button("📅 월간리포트", use_container_width=True):
            st.session_state.page = 'summaryReport'
            st.session_state.reportFlag = 'M'
            st.rerun()

    st.title(f"📊 {user_id}님의 수면 분석 리포트")
    # 포즈 클래스 선택
    if reportFlag == 'M':
        years = list(range(2025, 2031))
        months = list(range(1, 13))
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            year = st.selectbox("연도", years, index=years.index(date.today().year))
        with col2:
            month = st.selectbox("월", months, index=date.today().month - 1)
        with col3:
            pose_chart = ['1: 월간heatmap', '2: 월간pie', '3: 월간bar', '4: 연간_미정']
            selected_chart = st.selectbox("그래프 선택", pose_chart, index=0)
        selected_date = f"{year}{month:02d}"
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            d = st.date_input("날짜 선택", value=date.today())
        with col2:
            pose_chart = ['0: 일간gantt']
            selected_chart = st.selectbox("그래프 선택", pose_chart, index=0)
        selected_date =str(d)

    # 필터링 적용
    if selected_chart == "0: 일간gantt":
        ganttchart(user_id, selected_date)
    elif( selected_chart == "1: 월간heatmap"):
        heatmapChart(user_id, selected_date)
    elif( selected_chart == '2: 월간pie'):
        pieChart(user_id, selected_date)
    elif( selected_chart == '3: 월간bar'):
        barChart(user_id, selected_date)
    elif( selected_chart == '4: 연간_미정'):
        exit()

    st.markdown("---")
    if st.button("🏠 모니터링 화면으로", use_container_width=True):
        st.session_state.page = 'monitor'
        st.rerun()

