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

POSE_NAME_MAP = {
    0: "올바른 자세",
    1: "옆으로 누운 자세",
    2: "팔을 든 자세",
    3: "엎드린 자세",
}

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
당신은 **수면 자세 생활 코치**입니다.  
- 의료 전문가는 아니며, 진단이나 치료를 제공하지 않습니다.  
- 출력은 항상 한국어로 작성합니다.  
- 부드럽고 코칭 중심 톤으로, 사용자에게 불안을 주지 않습니다.

---

## 역할
1. 수면 자세 데이터를 이해하기 쉬운 방식으로 설명
2. 긍정적 피드백과 생활습관 개선 조언 제공
3. 신체 부담 가능성을 확률적·비의학적 표현으로 안내

## 제약
- 병명, 질병 진단, 약물, 시술, 치료 언급 금지
- “~할 수 있습니다”, “~할 가능성이 있습니다”와 같은 완화 표현 사용

---

# 사용자 수면 데이터 요약

### 총 수면 시간
- {total_sleep} 초

### 자세별 누적 시간
| 자세 | 시간(초) | 비율(%) |
|------|----------|----------|
| 올바른 자세 (권장) | {laying_time} | {laying_percent} |
| 옆으로 누운 자세 | {side_time} | {side_percent} |
| 팔을 든 자세 | {hand_up_time} | {hand_up_percent} |
| 엎드린 자세 | {back_time} | {back_percent} |

### 가장 시간이 긴 자세 
- {longist_nm}

---

## 평가 기준
- 올바른 자세 비율 ≥ 90% → 긍정적 평가 + 유지 팁
- 올바른 자세 비율 < 90% → 가장 시간이 긴 자세 중심 조언

---

## 요청 사항
1. 전체 수면 자세 상태 평가
2. 상태가 좋은 경우:
   - 격려 메시지
   - 현재 습관 유지 팁
3. 상태가 부족한 경우:
   - 가장 시간이 긴 자세 선택
   - 부담 가능 부위와 이유 설명
   - 비의료적 개선 방법 제안 (스트레칭, 수면 습관, 베개 활용 등)

---

## 출력 포맷 예시

### [전체 평가]
- 올바른 자세 90% 이상:
  > 오늘은 올바른 수면 자세로 대부분의 시간을 보내셨습니다.  
  > 전반적으로 잘 관리된 수면 패턴입니다.
- 올바른 자세 90% 미만:
  > n시간 동안 (가장 시간이 긴 자세)가 비교적 길게 유지되었습니다.

### [설명]
- 올바른 자세 90% 이상: 수면 중 신체 정렬이 안정적으로 유지되어 근육과 관절이 충분히 이완되었을 수 있습니다.
- 올바른 자세 90% 미만:  
  - 주요 자세: (가장 시간이 긴 자세)  
  - 부담 가능 부위: 목, 어깨, 골반  
  - 이유: 한쪽 체중 집중으로 신체 균형이 깨질 수 있습니다.

### [자세별 피드백] (90% 미만 시)
- 반복 시 기상 후 근육 긴장이나 뻐근함으로 이어질 수 있습니다.

### [생활습관 개선 제안] (90% 미만 시)
- 무릎 사이에 베개 사용하여 골반 정렬 보완  
- 기상 후 목·어깨 스트레칭 권장  
- 수면 중 자세가 한쪽으로 고정되지 않도록 환경 조정  

---

※ 본 내용은 생활습관 개선 참고용이며 의료적 조언이 아닙니다. (필수 기재)
"""


prompt = ChatPromptTemplate.from_template(prompt_template)
parser = StrOutputParser()
chain = prompt | llm_low | parser


from datetime import datetime
from collections import defaultdict

def calculate_pose_durations(pose_data):
    """
    pose_data: List[(pose_class, st_dt, ed_dt)] 또는 List[dict]
    return:
        total_time_sec: float
        pose_time_map: Dict[int, float]
    """
    total_time_sec = 0.0
    pose_time_map = defaultdict(float)

    for row in pose_data:
        # row가 tuple인지 dict인지 처리
        if isinstance(row, dict):
            pose_class = row.get('pose_class')
            st_dt = row.get('st_dt')
            ed_dt = row.get('ed_dt')
        else:
            pose_class, st_dt, ed_dt = row

        # None 건너뛰기
        if st_dt is None or ed_dt is None:
            continue

        # 문자열이면 datetime으로 변환
        if isinstance(st_dt, str):
            st_dt = datetime.fromisoformat(st_dt)
        if isinstance(ed_dt, str):
            ed_dt = datetime.fromisoformat(ed_dt)

        # 지속시간 계산
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
    try:
        # 1. 입력 데이터 유효성 검사
        if not data or len(data) == 0 or data[0] is None:
            return "수면 데이터가 없어 AI 분석을 제공할 수 없습니다."

        pose_data = data[0]
        if len(pose_data) == 0:
            return "분석 가능한 자세 데이터가 없습니다."

        # 2. 시간 계산
        total_time_sec, pose_time_map = calculate_pose_durations(pose_data)

        if total_time_sec <= 0:
            return "유효한 수면 시간이 부족하여 분석이 어렵습니다."

        pose_percentages = calculate_pose_percentages(total_time_sec, pose_time_map)
        print("pose_time_map:", pose_time_map)
        print("pose_percentages:", pose_percentages)
        
        # pose_time_map과 pose_percentages에서 key를 str -> int 변환
        pose_time_map = {int(k): v for k, v in pose_time_map.items()}
        pose_percentages = {int(k): v for k, v in pose_percentages.items()}

        valid_pose_time_map = {k: v for k, v in pose_time_map.items() if v > 0}

        if valid_pose_time_map:
            longest_pose_id = max(valid_pose_time_map, key=valid_pose_time_map.get)
            longest_pose_name = POSE_NAME_MAP.get(longest_pose_id, "알 수 없음")
        else:
            longest_pose_name = "분석 불가"

        input_data = {
            "total_sleep": total_time_sec,

            "laying_time": pose_time_map.get(0, 0.0),
            "laying_percent": pose_percentages.get(0, 0.0),

            "side_time": pose_time_map.get(1, 0.0),
            "side_percent": pose_percentages.get(1, 0.0),

            "hand_up_time": pose_time_map.get(2, 0.0),
            "hand_up_percent": pose_percentages.get(2, 0.0),

            "back_time": pose_time_map.get(3, 0.0),
            "back_percent": pose_percentages.get(3, 0.0),

            "longist_nm": longest_pose_name
        }

        # 4. LLM 호출
        result = chain.invoke(input_data)

        if not result:
            return "AI 분석 결과를 생성하지 못했습니다."

        return result

    except Exception as e:
        # 5. 전체 예외 캐치 (로그용)
        print(f"[LLM ERROR] {e}")
        return "AI 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
