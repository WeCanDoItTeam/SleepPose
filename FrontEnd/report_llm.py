import os
from dotenv import load_dotenv
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#======== [ 상수 선언 ] ========
MODEL_GPT_3_5_TURBO = "gpt-3.5-turbo" # 속도 낼 때. 무지 싸다
MODEL_GPT_4O_MINI = "gpt-4o-mini" # 테스트용. 싸다
MODEL_GPT_4 = "gpt-4" # 실전용. 비싸다

#======== [ 환경변수 가져오기 (.env) ] ========

load_dotenv()

#======== [ OpenAI ] ========

# Open API Key 불러오기
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

# LLM 가져오기
llm_very_low = ChatOpenAI(model=MODEL_GPT_3_5_TURBO, temperature=0, openai_api_key=openai_api_key)
llm_low = ChatOpenAI(model=MODEL_GPT_4O_MINI, temperature=0, openai_api_key=openai_api_key)
llm_high = ChatOpenAI(model=MODEL_GPT_4, temperature=0, openai_api_key=openai_api_key)

prompt_template = """ 
당신은 의료 전문가가 아닌 수면 자세 생활 코치입니다.
의학적 진단이나 치료를 제공하지 않습니다.
출력은 항상 한국어로 작성합니다.
전문적인 어조이되, 사용자에게 불안을 주지 않는 코칭 톤을 유지합니다.

역할:
- 수면 자세 데이터에 대해 이해하기 쉬운 설명 제공
- 긍정적 피드백과 생활습관 개선 조언 제공
- 신체 부담 가능성을 확률적·비의학적 표현으로 설명

제약:
- 병명, 질병 진단, 의학적 치료를 직접적으로 언급하지 말 것
- 약물, 시술, 병원 치료를 권유하지 말 것
- “~할 수 있습니다”, “~할 가능성이 있습니다”와 같은 완화된 표현 사용

아래는 한 사용자의 수면 자세 요약 데이터입니다.
이 데이터는 관찰 기반 정보이며, 의료적 진단 목적이 아닙니다.

[수면 요약]
- 총 수면 시간: {total_sleep}

[자세별 누적 시간]
- 올바른 자세(권장 자세): {laying_time} ({laying_percent}%)
- 옆으로 누운 자세: {side_time} ({side_percent}%)
- 팔을 든 자세: {hand_up_time} ({hand_up_percent}%)
- 엎드린 자세: {back_time} ({back_percent}%)

[평가 기준]
- 올바른 자세 비율이 90% 이상일 경우 → 긍정적 평가 및 유지 팁 제공
- 90% 미만일 경우 → 가장 많이 나타난 비권장 자세 중심으로 조언 제공

[요청 사항]
1. 위 기준에 따라 전체 수면 자세 상태를 평가하세요.
2. 상태가 좋은 경우:
- 격려 메시지
- 현재 습관을 유지하기 위한 간단한 팁 제공
3. 상태가 부족한 경우:
- 비율이 높은 비권장 자세를 하나 이상 선택
- 해당 자세로 인해 부담이 갈 수 있는 신체 부위 설명
- 왜 장시간 유지 시 불편감이 생길 수 있는지 설명
- 비의료적 개선 방법 제안 (스트레칭, 수면 습관, 베개 활용 등)
4. 전체 톤은 부드럽고 코칭 중심으로 작성하세요.

[제약 조건]
- 질병명 진단 금지
- 치료·약물·의학적 처치 언급 금지
- 구조화된 형식으로 출력


# 중요: 출력 포맷
[전체 평가]
출력 예시 (바른 자세 90% 이상):
오늘은 올바른 수면 자세로 대부분의 시간을 보내셨습니다.
전반적으로 잘 관리된 수면 패턴입니다.

출력 예시 (바른 자세 90% 미만):
n시간 동안 (가장 오래 유지된 비권장 자세 명)가 비교적 길게 유지되었습니다.

[설명 또는 격려]
출력 예시 (바른 자세 90% 이상):
수면 중 신체 정렬이 비교적 안정적으로 유지되어,
근육과 관절이 충분히 이완될 수 있었던 것으로 보입니다.

출력 예시 (바른 자세 90% 미만):
- 주요 자세: 옆으로 눕기
- 부담 가능 부위: 목, 어깨, 골반
- 이유: 한쪽으로 체중이 집중되면서 신체 균형이 깨질 수 있습니다

[자세별 피드백] (바른 자세 90% 미만 시)
출력 예시 (바른 자세 90% 미만):
이러한 자세가 반복될 경우,
기상 시 몸의 뻐근함이나 근육 긴장으로 이어질 가능성이 있습니다.

[유지 팁] (바른 자세 90% 이상 시)
출력 예시 (바른 자세 90% 이상):
- 현재 사용 중인 베개 높이를 유지해 주세요
- 취침 전 가벼운 목·어깨 스트레칭을 해보세요
- 수면 환경과 취침 시간을 일정하게 유지하는 것도 도움이 됩니다

[생활습관 개선 제안] (바른 자세 90% 미만 시)
출력 예시 (바른 자세 90% 미만):
- 무릎 사이에 베개를 끼워 골반 정렬을 보완해 보세요
- 기상 후 목과 어깨를 부드럽게 풀어주는 스트레칭을 권장합니다
- 수면 중 자세가 한쪽으로 고정되지 않도록 환경을 조정해 보세요

※ 본 내용은 생활습관 개선을 위한 참고 정보이며 의료적 조언이 아닙니다. (필수 기재)
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
parser = StrOutputParser()
chain = prompt | llm_low | parser


def calculate_pose_durations(pose_data):
    """
    pose_data: List[(pose_class, st_dt, ed_dt)]
    return:
        total_time_sec: float
        pose_time_map: Dict[int, float]
    """
    total_time_sec = 0.0
    pose_time_map = defaultdict(float)

    for pose_class, st_dt, ed_dt in pose_data:
        if st_dt is None or ed_dt is None:
            continue

        duration = (ed_dt - st_dt).total_seconds()
        if duration <= 0:
            continue

        total_time_sec += duration
        pose_time_map[pose_class] += duration

    return total_time_sec, dict(pose_time_map)

def calculate_pose_percentages(total_time_sec, pose_durations):
    """
    total_time_sec: float
    pose_durations: Dict[int, float]
    return: Dict[int, float]  # percentage
    """
    if total_time_sec <= 0:
        return {pose: 0.0 for pose in pose_durations}

    pose_percentages = {}
    for pose, sec in pose_durations.items():
        pose_percentages[pose] = (sec / total_time_sec) * 100

    return pose_percentages

def get_llm(data):
    if data is None: 
        return

    total_time_sec, pose_time_map = calculate_pose_durations(data[0])
    pose_percentages = calculate_pose_percentages(total_time_sec, pose_time_map)
    input_data = {"total_sleep": total_time_sec,
                  "laying_time" : pose_time_map[0],
                  "laying_percent" : pose_percentages[0],
                  "side_time" : pose_time_map[1],
                  "side_percent" : pose_percentages[1],
                  "hand_up_time" : pose_time_map[2],
                  "hand_up_percent" : pose_percentages[2],
                  "back_time" : pose_time_map[3],
                  "back_percent" : pose_percentages[3]}
    

    result = chain.invoke(input_data)

    return result