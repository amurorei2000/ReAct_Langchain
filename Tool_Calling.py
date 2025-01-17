from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_anthropic import ChatAnthropic

load_dotenv()


@tool
def multiply(x: float, y: float) -> float:
    """x와 y의 곱셈 연산"""
    return x * y


if __name__ == "__main__":
    print("Hello Tool Calling")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tools = [TavilySearchResults(), multiply]
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    response = agent_executor.invoke(
        {
            "input": "두바이의 날씨는 지금 어때? 대한민국의 현재 날씨와 비교해 줘. 온도는 섭씨로 출력해 줘"
        }
    )

    print(response["output"])
