from typing import Any
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    # LLM의 동작이 시작할 때
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        print(f"***LLM에 들어가는 프롬프트:***\n{prompts[0]}")
        print("********")

    # LLM의 동작이 끝났을 때
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        print(f"***LLM의 응답:***\n{response.generations[0][0].text}")
        print("********")
