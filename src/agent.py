"""메인 에이전트 그래프: 하위 Agent와 Supervisor를 조립한다(build_app)."""

from contextlib import AsyncExitStack
from typing import Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool

from src.config import llm, today
from src.guardrails import REFUSAL_MESSAGES, outfit_lines
from src.middleware import LoggingMiddleware, TextOnlyHistoryMiddleware, ToolFirstMiddleware
from src.tools import AGENT_TOOLS, HITL_TOOLS, open_tools, tools_for

# 테스트테스트

RECURSION_LIMIT = 25
TOOL_CALL_LIMIT = 10
STYLIST_TEMPERATURE = 0.3  # 후보 안에서의 조합과 이유 작성. 판정·라우팅이 아니라서 0보다 약간 높인다

STYLIST_PROMPT = """너는 오늘 입을 코디를 추천하는 stylist_agent다. 오늘 코디 추천, 오늘 날씨(기온·비·바람), 일정에 맞는 옷만 담당한다.
"""

LOG_PROMPT = """너는 착용 기록과 피드백을 맡는 log_agent다. 입은 옷 기록, 지난 착용 조회, 추웠다·더웠다 피드백, 추위 민감도·선호 스타일·지역 변경만 담당한다.
- 날짜는 사용자가 말한 표현 그대로("오늘", "어제", "이번 주 수요일") 도구에 넘기고 직접 계산하지 않는다. 날짜를 말하지 않으면 "오늘"이다. 지난 기록을 상황(비 온 날, 데이트 날)·옷·피드백으로 물으면 search_wear_history(조건은 rain·event_type·item_name·direction에)로, 최근 며칠을 날짜순으로 물으면 get_recent_wear로 찾고, 결과에 있는 기록만 근거로 답한다(없으면 없다고 안내).
- "N번 입을게/입었어"는 get_recommendation으로 그날 N번 후보의 item_ids를 찾아 confirm_wear를 호출한다. 그 번호가 없으면 있는 번호를 알려 주고 호출하지 않는다.
- 춥다·덥다·불편했다는 말은 record_feedback으로 기록하고, 결과에 suggest_profile_change가 있으면 따로 묻지 말고 곧바로 update_profile(field, to)을 호출한다(확인은 시스템이 받는다). "나 추위 많이 타"처럼 체질을 직접 말하면 get_profile로 현재 값을 보고 한 단계만 update_profile을 호출한다. 사용자가 get_profile의 change_rules를 벗어난 값을 직접 말하면(예: "-5로 바꿔줘") update_profile을 부르지 말고(한 단계로 바꿔 호출하지도 않는다) 바꿀 수 있는 범위가 -2~+2이고 한 번에 한 단계씩만 바뀐다고 안내한다.
- 기록·변경 결과는 도구 결과의 실제 값(confirm_wear의 recorded_items, update_profile의 to)으로만 짧게 안내한다. 사용자가 승인 단계에서 수정했을 수 있으니 처음 요청한 값이나 추천 번호의 옷을 다시 말하지 않고, 수정됐으면 "N번"이라고 부르지 않는다. 거절되거나 실패하면 바꾸지 않았다고 안내한다. 담당 범위 밖 질문에는 "담당 범위가 아닙니다"라고만 답한다."""

# HITL에서 사람이 인자를 고쳐 실행했을 때 도구 결과 앞에 붙는 안내(원래 값으로 안내하는 실수를 막는다)
EDIT_NOTICE = "사용자가 승인 단계에서 값을 수정해 실행했습니다. 처음 요청한 값이 아니라 아래 도구 결과의 실제 값으로만 안내하세요."

CLOSET_PROMPT = """너는 옷장과 일정을 관리하는 closet_agent다. 옷 추가, 옷 목록, 세탁 상태 변경, 캘린더 일정 추가만 담당한다.
- 옷 추가: 종류·두께(1~5)·색(영문 소문자)·격식·방수는 이름에서 추정해 add_clothing을 호출하고, 저장된 값을 그대로 보여 주며 틀리면 알려 달라고 한다.
- 세탁 상태: "세탁 넣었어/빨래통에 넣었어"는 laundry, "세탁 다 했어/빨래 끝났어/다 말랐어"는 available이다. id를 모르면 list_clothes로 이름에 맞는 옷의 id를 찾은 뒤 update_status를 호출한다.
- 요청마다 이번에 도구를 새로 호출하고 그 결과로만 답한다. 앞선 대화에서 같은 옷을 다뤘어도 "바꿨다", "추가했다", "저장했다"는 이번에 호출한 도구 결과가 있을 때만 말한다.
- 일정 추가: 날짜는 사용자가 말한 표현 그대로 넘기고, 제목(무슨 일정인지)이나 날짜가 빠졌으면 호출하지 말고 되묻는다. 저장된 날짜·시간·종류를 그대로 보여 준다.
- 도구 결과의 실제 값으로만 안내한다. 담당 범위 밖 질문에는 "담당 범위가 아닙니다"라고만 답한다."""

GENERAL_PROMPT = f"""너는 인사와 범위 밖 질문을 맡는 general_agent다. 도구는 없다.
- 인사·잡담에는 한두 문장으로 짧고 친절하게 답하고, 할 수 있는 일(오늘 코디 추천, 옷장·일정 관리, 착용 기록·피드백)을 안내한다.
- 옷·날씨(기온·비·바람)·일정과 관계없는 질문(미세먼지·황사·자외선·꽃가루 같은 다른 날씨 정보 포함)에는 정보를 지어내지 말고 아래 문장만 그대로 답한다.
{REFUSAL_MESSAGES["off_topic"]}"""

SUPERVISOR_PROMPT = """너는 아침 옷 추천 서비스의 작업 분배자(Supervisor)다. 직접 답하거나 되묻지 말고 항상 담당 Agent에게 맡긴다(정보가 빠진 요청도 담당 Agent가 되묻는다).
[배분 기준]
- 코디, 추천, 오늘 뭐 입지, 날씨(기온·비·바람), 일정에 맞는 옷, 특정 옷을 입어도 되는지, 앞선 추천에 대한 질문 → stylist_agent
- 옷 추가, 옷 목록, 세탁, 옷 상태, 일정 추가 → closet_agent
- 입은 옷("N번 입을게/입었어"), 착용 기록, 지난 기록 찾기(…때 뭐 입었지, 그 옷 입은 날 어땠지), 추웠다·더웠다 피드백, 추위 민감도, 선호 스타일, 지역 변경 → log_agent
- 인사, 잡담, 미세먼지·황사·자외선 등 기온·비·바람 밖의 날씨, 그 밖의 모든 질문 → general_agent
[규칙]
- 피드백·기록과 추천이 섞인 질문("어제 좀 추웠어. 오늘은 뭐 입지?")은 log_agent에게 먼저, 그다음 stylist_agent에게 맡긴다.
- 하위 Agent가 답하면 요약하거나 고쳐 쓰지 말고 forward_message(from_agent=마지막으로 답한 Agent 이름)로 원문을 그대로 전달하고 끝낸다. 복합 질문은 마지막 stylist_agent 답을 전달한다."""


_library_forward = create_forward_message_tool("supervisor")


@tool("forward_message", description=_library_forward.description)
def forward_message(from_agent: str, state: Annotated[dict, InjectedState]):
    """하위 Agent의 답을 그대로 사용자에게 넘긴다. 단, 그 Agent가 이번 요청(마지막 사용자 발화 뒤)에 답했을 때만.

    라이브러리 도구는 대화 전체에서 그 Agent의 마지막 답을 찾으므로, Supervisor가 맡기지 않고 바로 부르면
    앞선 턴의 추천(다른 날씨)을 이번 답처럼 넘긴다. 그때는 넘기지 않고 먼저 맡기라는 안내를 돌려준다.
    """
    messages = state["messages"]
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    answered = any(isinstance(m, AIMessage) and (m.name or "").lower() == from_agent.lower() for m in messages[start + 1:])
    if not answered:
        return (f"{from_agent}는 이번 요청에 아직 답하지 않았습니다. 앞선 답을 다시 넘기지 말고 "
                f"transfer_to_{from_agent}로 먼저 맡긴 뒤 그 답을 forward_message로 넘기세요.")
    return _library_forward.func(from_agent, state)


def thread_config(user_id: str = "me") -> dict:
    """그날의 대화 스레드 설정. thread_id는 {user_id}:{오늘 YYYY-MM-DD}이고 반복 상한을 함께 둔다."""
    return {"configurable": {"thread_id": f"{user_id}:{today().isoformat()}"}, "recursion_limit": RECURSION_LIMIT}


async def build_app(checkpointer, settings: dict, stack: AsyncExitStack):
    """MCP 서버를 띄우고 하위 Agent와 Supervisor를 조립해 컴파일한 앱을 돌려준다.

    settings는 서버 env로 넘길 실행 설정이다. MCP 세션은 stack이 닫힐 때 함께 닫히므로,
    설정이 바뀌면 이전 stack을 닫고 새 stack으로 다시 부른다(체크포인터는 그대로 넘겨 쓴다).
    """
    tools = await open_tools(stack, settings)
    stylist = create_agent(
        model=llm(STYLIST_TEMPERATURE),
        tools=tools_for("stylist_agent", tools),
        system_prompt=STYLIST_PROMPT,
        middleware=[ToolCallLimitMiddleware(thread_limit=TOOL_CALL_LIMIT),
                    ToolFirstMiddleware(AGENT_TOOLS["stylist_agent"], only_if=lambda text: bool(outfit_lines(text))),
                    LoggingMiddleware("stylist_agent")],
        name="stylist_agent",
    )
    closet = create_agent(
        model=llm(0),
        tools=tools_for("closet_agent", tools),
        system_prompt=CLOSET_PROMPT,
        middleware=[ToolCallLimitMiddleware(thread_limit=TOOL_CALL_LIMIT),
                    ToolFirstMiddleware(AGENT_TOOLS["closet_agent"]), LoggingMiddleware("closet_agent")],
        name="closet_agent",
    )
    log = create_agent(
        model=llm(0),
        tools=tools_for("log_agent", tools),
        system_prompt=LOG_PROMPT,
        middleware=[
            ToolCallLimitMiddleware(thread_limit=TOOL_CALL_LIMIT),
            ToolFirstMiddleware(AGENT_TOOLS["log_agent"]),
            HumanInTheLoopMiddleware(
                interrupt_on={name: {"allowed_decisions": ["approve", "edit", "reject"]} for name in HITL_TOOLS},
                edit_notice=EDIT_NOTICE,
            ),
            LoggingMiddleware("log_agent"),
        ],
        name="log_agent",
    )
    general = create_agent(
        model=llm(0),
        tools=[],
        system_prompt=GENERAL_PROMPT,
        middleware=[ToolCallLimitMiddleware(thread_limit=TOOL_CALL_LIMIT), TextOnlyHistoryMiddleware(),
                    LoggingMiddleware("general_agent")],
        name="general_agent",
    )
    supervisor = create_supervisor(
        [stylist, closet, log, general],
        model=llm(0),
        prompt=SUPERVISOR_PROMPT,
        tools=[forward_message],
        output_mode="last_message",
    )
    return supervisor.compile(checkpointer=checkpointer)
