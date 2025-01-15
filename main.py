from dotenv import load_dotenv
from langchain.agents import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import Union, List
from langchain.tools import Tool
from langchain.agents.format_scratchpad import format_log_to_str
from callbacks import AgentCallbackHandler

# from langchain import hub

load_dotenv()


# 랭체인 툴 함수로 등록
@tool
def get_text_length(text: str) -> int:
    """문자열의 글자 수를 반환합니다."""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)


# 툴 목록에서 툴 이름과 동일한 것이 있는지 찾는 함수
def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool name with {tool_name} not found.")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")

    # 툴 목록
    tools = [get_text_length]

    # hub.pull("hwchase17/react")
    template = """
    가능한  최선으로 다음 질문에 답합니다. 당신은 다음 도구에 접근할 수 있습니다.

    {tools}

    다음 형식을 사용하세요:

    Question: 반드시 답변해야 하는 입력된 질문
    Thougth: 무엇을 해야할 지에 대해 항상 생각해야 합니다.
    Action: [{tool_names}] 중에 하나로서 취할 행동
    Action Input: Action의 입력 값
    Observation: Action의 결과 값
    ... (이 Thought/Action/Action Input/Observation은 N번 반복할 수 있습니다.)
    Thought: 나는 지금 최종 답변을 안다.
    Final Answer: 입력된 질문 원본에 대한 최종 답변

    시작!

    Question: {input}
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(
        temperature=0,
        stop=["\nObservation"],
        callbacks=[AgentCallbackHandler()],
        model="gpt-4o-mini",
    )
    intermediate_steps = []

    # LCEL
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""

    # 에이전트의 현재 단계가 종료 단계가 아니면 계속 반복
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "Dog의 글자 수는 몇 개인가요?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        # print(agent_step)

        # 에이전트 액션 로그 확인
        if isinstance(agent_step, AgentAction):
            # LLM의 답변에서 사용할 tool 이름과 tool에 들어갈 인자 값을 가져온다.
            tool_name = agent_step.tool
            tool_input = agent_step.tool_input

            # tool 이름에서 함수를 찾고 매개변수에 tool_input 인자 값을 넣는다.
            tool_to_use = find_tool_by_name(tools, tool_name)
            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")

            # 현재 에이전트 단계와 결과 값을 저장한다.
            intermediate_steps.append((agent_step, str(observation)))

    # 에이전트 피니쉬 로그 확인
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
