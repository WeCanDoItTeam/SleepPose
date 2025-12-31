import streamlit as st
import pandas as pd
from db_utils import get_sleep_day, get_sleep_month
import plotly.graph_objects as go
import plotly.express as px
import calendar
import uuid
from datetime import datetime, timedelta, date

class ColorPalette:
    laying = '#FCA47C'
    side = '#F9D779'
    hand_up = '#23CED9'
    back = "#A1CCA6"
    others = "#097C87"

    snoring = '#A4D4D4'
    bruxism = "#ECA0B0"
    noise = "#847E7C"

color = ColorPalette()

def startEndDate(yyyymm):
    year = int(yyyymm[:4])
    month = int(yyyymm[4:])
    start_date = date(year, month, 1)  # 2025-03-01
    last_day = calendar.monthrange(year, month)[1]
    end_date = date(year, month, last_day)
    return (start_date, end_date)


def addDate(yyyymmdd):
    d = datetime.strptime(yyyymmdd, "%Y-%m-%d").date()
    next_day = d + timedelta(days=1)
    return str(next_day)


def getUuid():
    chart_key = str(uuid.uuid4())
    return chart_key


def ganttchart(user_id, st_dt):
    ed_dt = addDate(st_dt)
    # 1. 데이터 가져오기
    (pose_data, audio_data) = get_sleep_day(user_id, st_dt, ed_dt)
    pose_df = pd.DataFrame(pose_data)
    if pose_df.empty:
        st.markdown("수면포즈 데이터가 없습니다")
        return
    st.markdown("### 그래프 필터링 옵션")
    pose_df['st_dt'] = pd.to_datetime(pose_df['st_dt'])
    pose_df['ed_dt'] = pd.to_datetime(pose_df['ed_dt'])
    min_t = pose_df['st_dt'].min()
    max_t = pose_df['ed_dt'].max()

    pose_df['pose_class'] = pose_df['pose_class'].astype(str)
    
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
    view_pose_df = pose_df[mask].copy()

    fig = px.timeline(
        view_pose_df,
        x_start='st_dt',
        x_end='ed_dt',
        y='pose_class',
        color='pose_class',
        labels={
            'st_dt': '시작 시간',
            'ed_dt': '종료 시간',
            'pose_class': '수면 자세'
        },
        color_discrete_map={
            '바로 누운 자세': color.laying,
            '옆으로 누워 자기': color.side,
            '팔든 자세': color.hand_up,
            '엎드린 자세': color.back,
            '기타': color.others,
        },
        title="수면포즈 시분초 Gantt 차트"
    )

    fig.update_layout(
        xaxis_title='시간대(시)',
        yaxis_title='',
        showlegend=False,
        dragmode=False,              # ✅ 드래그 차단
        xaxis=dict(
            type='date',
            tickformat='%H:%M',          # ✅ 분 단위 표시
            dtick=10 * 60 * 1000,        # ✅ 10분 간격
            fixedrange=True,         # ✅ 확대/이동 차단
            rangeslider=dict(visible=False)  # ✅ 위에 생기는 스크롤 제거
        ),
        yaxis=dict(fixedrange=True)  # ✅ Y축도 고정
    )

    # 🔥 Plotly 인터랙션 완전 차단
    st.plotly_chart(
        fig,
        key=getUuid(),
        use_container_width=True,
        config={
            "scrollZoom": False,     # ✅ 마우스 휠 확대 차단
            "displayModeBar": False, # (선택) 우측 툴바 제거
            "doubleClick": False     # 더블클릭 확대 차단
        }
    )
    
    audio_df = pd.DataFrame(audio_data)
    if audio_df.empty:
        st.markdown("코골이,이갈이 데이터가 없습니다")
        return

    mask = (audio_df['st_dt'] >= start_t) & (audio_df['st_dt'] <= end_t)
    view_audio_df = audio_df[mask].copy()

    fig = px.timeline(
        view_audio_df,
        x_start='st_dt',
        x_end='ed_dt',
        y='audio_class',
        color='audio_class',
        labels={
            'st_dt': '시작 시간',
            'ed_dt': '종료 시간',
            'audio_class': '음성 분류'
        },
        color_discrete_map={
            '기타': color.noise,
            '코골이': color.snoring,
            '이갈이': color.bruxism
        },
        title="코골이·이갈이 시분초 Gantt 차트"
    )

    fig.update_layout(
        xaxis_title='시간대(시)',
        yaxis_title='',
        showlegend=False,
        dragmode=False,  # ✅ 드래그 줌 차단
        xaxis=dict(
            type='date',
            tickformat='%H:%M',          # ✅ 분 단위
            dtick=10 * 60 * 1000,        # ✅ 10분 간격
            fixedrange=True,                 # ✅ 확대/이동 차단
            rangeslider=dict(visible=False)  # ✅ 상단 스크롤 제거
        ),
        yaxis=dict(fixedrange=True)          # ✅ Y축 고정
    )

    st.plotly_chart(
        fig,
        key=getUuid(),
        use_container_width=True,
        config={
            "scrollZoom": False,     # ✅ 마우스 휠 확대 차단
            "displayModeBar": False, # (선택) 우측 툴바 제거
            "doubleClick": False     # 더블클릭 확대 차단
        }
    )



def heatmapChart(user_id, st_dt):
    # 1달간 시간대별 자세 소요시간 집계
    (start_date, end_date) = startEndDate(st_dt)
    (pose_data, audio_data) = get_sleep_month(user_id, str(start_date), str(end_date))

    pose_df = pd.DataFrame(pose_data)
    if pose_df.empty:
        st.markdown("## 데이터가 없습니다")
        return

    pose_df['hour_slot'] = pose_df['hour_slot'].astype(int)  # 시간대별
    pose_df['pose_class'] = pose_df['pose_class'].astype(str)
    pose_df['minutes'] = pose_df['minutes'].astype(float)  # 소요시간(분)
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
        xaxis_title='시간대(시)',  # 12, 13, 14 ...
        yaxis_title='포즈 클래스',  # 0,1,2,3,4
        coloraxis_colorbar_title='시간합(분)'  # 색바 라벨
    )
    chartkey = getUuid()
    st.plotly_chart(fig, width='stretch', key=chartkey)

    audio_df = pd.DataFrame(audio_data)
    if audio_df.empty:
        st.markdown("## 데이터가 없습니다")
        return

    audio_df['hour_slot'] = audio_df['hour_slot'].astype(int)  # 시간대별
    audio_df['audio_class'] = audio_df['audio_class'].astype(str)
    audio_df['minutes'] = audio_df['minutes'].astype(float)  # 소요시간(분)
    audio_df['audio_nm'] = audio_df['audio_nm'].astype(str)

    fig = px.density_heatmap(
        audio_df,
        x='hour_slot',
        y='audio_nm',
        z='minutes',
        color_continuous_scale='Viridis',
        title='한달간 시간대별 이갈이,코골이 소요시간 Heatmap(분)'
    )

    fig.update_layout(
        xaxis_title='시간대(시)',  # 12, 13, 14 ...
        yaxis_title='이갈이,코골이',  # 0,1,2,3,4
        coloraxis_colorbar_title='시간합(분)'  # 색바 라벨
    )
    chartkey = getUuid()
    st.plotly_chart(fig, width='stretch', key=chartkey)


def pieChart(user_id, st_dt):
    (start_date, end_date) = startEndDate(st_dt)
    (pose_data, audio_data) = get_sleep_month(user_id, str(start_date), str(end_date))

    pose_df = pd.DataFrame(pose_data)
    if pose_df.empty:
        st.markdown("## 데이터가 없습니다")
        return

    pose_df['hour_slot'] = pose_df['hour_slot'].astype(int)  # 시간대별
    pose_df['pose_class'] = pose_df['pose_class'].astype(str)
    pose_df['minutes'] = pose_df['minutes'].astype(float)  # 소요시간(분)
    pose_df['pose_nm'] = pose_df['pose_nm'].astype(str)

    labels = pose_df['pose_nm'].tolist()
    values = pose_df['minutes'].tolist()

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                                 insidetextorientation='radial',
                                 marker=dict(colors=[color.laying, color.side, color.hand_up, color.back, color.others]),
                                 hovertemplate='%{label}: %{value:.1f}분 (%{percent})<extra></extra>',
                                 )])
    fig.update_layout(
        title=dict(
            text="월간 자세분석(%)",
            x=0.1,  # 중앙
            # font=dict(size=24, color="darkblue")
        ),
        showlegend=False,  # 범례 제거
    )
    st.plotly_chart(fig, width='stretch', key=getUuid())

    audio_df = pd.DataFrame(audio_data)
    if audio_df.empty:
        st.markdown("## 데이터가 없습니다")
        return

    audio_df['hour_slot'] = audio_df['hour_slot'].astype(int)  # 시간대별
    audio_df['audio_class'] = audio_df['audio_class'].astype(str)
    audio_df['minutes'] = audio_df['minutes'].astype(float)  # 소요시간(분)
    audio_df['audio_nm'] = audio_df['audio_nm'].astype(str)

    labels = audio_df['audio_nm'].tolist()
    values = audio_df['minutes'].tolist()

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                                 insidetextorientation='radial',
                                 marker=dict(colors=[color.noise, color.snoring, color.bruxism]),
                                 hovertemplate='%{label}: %{value:.1f}분 (%{percent})<extra></extra>',
                                 )])
    fig.update_layout(
        title=dict(
            text="월간 이갈이,코골이 분석(%)",
            x=0.1,  # 중앙
            # font=dict(size=24, color="darkblue")
        ),
        showlegend=False,  # 범례 제거
    )
    st.plotly_chart(fig, width='stretch', key=getUuid())


def barChart_Day(user_id, st_dt):
    (start_date, end_date) = startEndDate(st_dt)
    (pose_data, audio_data) = get_sleep_month(user_id, str(start_date), str(end_date), gubun='%d')

    pose_df = pd.DataFrame(pose_data)

    if pose_df.empty:
        st.markdown("## 데이터가 없습니다")
        return

    pose_df['hour_slot'] = pose_df['hour_slot'].astype(int)  # 시간대별
    pose_df['pose_class'] = pose_df['pose_class'].astype(str)
    pose_df['minutes'] = pose_df['minutes'].astype(float)  # 소요시간(분)
    pose_df['pose_nm'] = pose_df['pose_nm'].astype(str)

    labels = pose_df['pose_nm'].tolist()
    values = pose_df['minutes'].tolist()

    long_df = px.data.medals_long()
    fig = px.bar(pose_df,
                 x="hour_slot",
                 y="minutes",
                 color="pose_nm",
                 barmode='stack',  # 병렬 막대
                 title="시간대별 포즈 소요시간(분)",
                 text="minutes",
                 labels={
                    'hour_slot': '시간대 별',
                    'minutes': '지속 시간(분)',
                    'pose_nm': '자세 명'
                    },
                 color_discrete_map={
                    '바로 누운 자세': color.laying,
                    '옆으로 누워 자기': color.side,
                    '팔든 자세': color.hand_up,
                    '엎드린 자세': color.back,
                    '기타': color.others,
                 },
                 )
    fig.update_layout(
        xaxis_title='일별',  # 12, 13, 14 ...
        yaxis_title='포즈 클래스',  # 0,1,2,3,4
        legend=dict(
            y=-0.5,
            x=0.5,
            xanchor="center",
            yanchor="top",
            orientation="h",  # 이 줄 추가: 수평 범례
        ),
        margin=dict(b=200),
        legend_title_text=''
    )
    # ✅ 막대 위에 분 단위 표시
    fig.update_traces(
        hovertemplate='%{x}일<br>%{y:.0f}분<br>%{fullData.name}',
        texttemplate='%{y:.0f}분',  # y값을 정수 분으로 표시
        textposition='inside'
    )
    st.plotly_chart(fig, width='stretch', key=getUuid())

    audio_df = pd.DataFrame(audio_data)

    if audio_df.empty:
        st.markdown("## 데이터가 없습니다")
        return

    audio_df['hour_slot'] = audio_df['hour_slot'].astype(int)  # 시간대별
    audio_df['audio_class'] = audio_df['audio_class'].astype(str)
    audio_df['minutes'] = audio_df['minutes'].astype(float)  # 소요시간(분)
    audio_df['audio_nm'] = audio_df['audio_nm'].astype(str)

    labels = audio_df['audio_nm'].tolist()
    values = audio_df['minutes'].tolist()

    long_df = px.data.medals_long()
    fig = px.bar(audio_df,
                 x="hour_slot",
                 y="minutes",
                 color="audio_nm",
                 barmode='stack',  # 병렬 막대
                 title="시간대별 이갈이,코골이 소요시간(분)",
                 text="minutes",
                 labels={
                    'hour_slot': '시간대 별',
                    'minutes': '지속 시간',
                    'audio_nm': '분류'
                 },
                 color_discrete_map={
                     '이갈이': color.bruxism,
                     '코골이': color.snoring,
                     '기타': color.noise
                 }
                 )
    fig.update_layout(
        xaxis_title='일별',  # 12, 13, 14 ...
        yaxis_title='이갈이,코골이 클래스',  # 0,1,2,3,4
        legend=dict(
            y=-0.5,
            x=0.5,
            xanchor="center",
            yanchor="top",
            orientation="h",  # 이 줄 추가: 수평 범례
        ),
        margin=dict(b=200),
        legend_title_text=''
    )
    # ✅ 막대 위에 분 단위 표시
    fig.update_traces(
        hovertemplate='%{x}일<br>%{y:.0f}분<br>%{fullData.name}',
        texttemplate='%{y:.0f}분',  # y값을 정수 분으로 표시
        textposition='inside'
    )
    st.plotly_chart(fig, width='stretch', key=getUuid())


def barChart_Hour(user_id, st_dt):
    (start_date, end_date) = startEndDate(st_dt)
    (pose_data, audio_data) = get_sleep_month(user_id, str(start_date), str(end_date))
    pose_df = pd.DataFrame(pose_data)

    if pose_df.empty:
        st.markdown("## 데이터가 없습니다")
        return

    pose_df['hour_slot'] = pose_df['hour_slot'].astype(int)  # 시간대별
    pose_df['pose_class'] = pose_df['pose_class'].astype(str)
    pose_df['minutes'] = pose_df['minutes'].astype(float)  # 소요시간(분)
    pose_df['pose_nm'] = pose_df['pose_nm'].astype(str)

    labels = pose_df['pose_nm'].tolist()
    values = pose_df['minutes'].tolist()

    long_df = px.data.medals_long()
    fig = px.bar(pose_df,
                 x="hour_slot",
                 y="minutes",
                 color="pose_nm",
                 barmode='stack',  # 병렬 막대
                 title="시간대별 포즈 소요시간(분)",
                 text="minutes",
                 labels={
                    'hour_slot': '시간대 별',
                    'minutes': '지속 시간',
                    'pose_nm': '자세 명'
                 },
                 color_discrete_map={
                     '바로 누운 자세': color.laying,
                     '옆으로 누워자기': color.side,
                     '팔든 자세': color.hand_up,
                     '엎드린 자세': color.back,
                     '기타': color.others
                 }
                 )
    fig.update_layout(
        xaxis_title='시간대(시)',  # 12, 13, 14 ...
        yaxis_title='포즈 클래스',  # 0,1,2,3,4
        legend=dict(
            y=-0.5,
            x=0.5,
            xanchor="center",
            yanchor="top",
            orientation="h",  # 이 줄 추가: 수평 범례
        ),
        margin=dict(b=200),
        legend_title_text = ''
    )
    fig.update_traces(
        hovertemplate='%{x}시<br>%{y:.0f}분<br>%{fullData.name}',
        texttemplate='%{y:.0f}분',  # y값을 정수 분으로 표시
        textposition='inside'
    )
    st.plotly_chart(fig, width='stretch', key=getUuid())

    audio_df = pd.DataFrame(audio_data)
    if audio_df.empty:
        st.markdown("## 데이터가 없습니다")
        return

    audio_df['hour_slot'] = audio_df['hour_slot'].astype(int)  # 시간대별
    audio_df['audio_class'] = audio_df['audio_class'].astype(str)
    audio_df['minutes'] = audio_df['minutes'].astype(float)  # 소요시간(분)
    audio_df['audio_nm'] = audio_df['audio_nm'].astype(str)

    labels = audio_df['audio_nm'].tolist()
    values = audio_df['minutes'].tolist()

    long_df = px.data.medals_long()
    fig = px.bar(audio_df,
                 x="hour_slot",
                 y="minutes",
                 color="audio_nm",
                 barmode='stack',  # 병렬 막대
                 title="시간대별 이갈이,코골이 소요시간(분)",
                 text="minutes",
                 color_discrete_map={
                     '이갈이': color.bruxism,
                     '코골이': color.snoring,
                     '기타': color.noise
                 },
                 )
    fig.update_layout(
        xaxis_title='시간대(시)',  # 12, 13, 14 ...
        yaxis_title='이갈이,코골이 클래스',  # 0,1,2,3,4
        legend=dict(
            y=-0.5,
            x=0.5,
            xanchor="center",
            yanchor="top",
            orientation="h",  # 이 줄 추가: 수평 범례
        ),
        margin=dict(b=200),
        legend_title_text=''
    )
    fig.update_traces(
        hovertemplate='%{x}시<br>%{y:.0f}분<br>%{fullData.name}',
        texttemplate='%{y:.0f}분',  # y값을 정수 분으로 표시
        textposition='inside'
    )
    st.plotly_chart(fig, width='stretch', key=getUuid())


def report_window():
    user_id = st.session_state.user_id
    reportFlag = st.session_state.reportFlag  # default:D
    selected_chart = None
    selected_date = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🕒일간 리포트", width='stretch'):
            st.session_state.page = 'summaryReport'
            st.session_state.reportFlag = 'D'
            st.rerun()
    with col2:
        if st.button("📅 월간리포트", width='stretch'):
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
            # pose_chart = ['1: 월간heatmap', '2: 월간_자세별', '3: 월간_시간별', '4: 월간_일자별']
            pose_chart = ['1: 월간_자세별', '2: 월간_시간별', '3: 월간_일자별']
            selected_chart = st.selectbox("그래프 선택", pose_chart, index=0)
        selected_date = f"{year}{month:02d}"
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            d = st.date_input("날짜 선택", value=date.today())
        with col2:
            pose_chart = ['0: 일간gantt']
            selected_chart = st.selectbox("그래프 선택", pose_chart, index=0)
        selected_date = str(d)

    # 필터링 적용
    if selected_chart == "0: 일간gantt":
        ganttchart(user_id, selected_date)
    # elif (selected_chart == "1: 월간heatmap"):
    #     heatmapChart(user_id, selected_date)
    elif (selected_chart == '1: 월간_자세별'):
        pieChart(user_id, selected_date)
    elif (selected_chart == '2: 월간_시간별'):
        barChart_Hour(user_id, selected_date)
    elif (selected_chart == '3: 월간_일자별'):
        barChart_Day(user_id, selected_date)
    st.markdown("---")
    if st.button("🏠 모니터링 화면으로", width='stretch'):
        st.session_state.page = 'monitor'
        st.rerun()

