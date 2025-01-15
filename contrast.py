from langchain.agents import initialize_agent, AgentType, tool, AgentExecutor
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """문자열의 글자 수를 반환합니다."""
    # print(f"get_text_length enter with {text=}")
    text = text.strip("\n").strip("'")
    return len(text)


if __name__ == "__main__":
    # LLM 생성 및 초기화
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
    )

    # 에이전트 생성 및 초기화
    agent_executor: AgentExecutor = initialize_agent(
        tools=[get_text_length],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    response: Dict[str, Any] = agent_executor.invoke(
        {"input": "Dog의 글자 수는 몇 개인가요?"}
    )

    print("=" * 50)
    print(response)
