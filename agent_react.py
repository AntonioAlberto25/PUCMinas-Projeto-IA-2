from __future__ import annotations

from langchain_core.messages import SystemMessage

try:
    from langchain.agents import create_agent as _create_agent

    HAS_LANGCHAIN_CREATE_AGENT = True
except Exception:
    from langgraph.prebuilt import create_react_agent as _create_react_agent

    HAS_LANGCHAIN_CREATE_AGENT = False

from providers import get_chat_model
from tools import generate_unit_tests


def create_agent():
    """Cria um agente ReAct focado somente em geracao de testes unitarios."""
    llm = get_chat_model(temperature=0)

    system_prompt = (
        "Voce e um engenheiro de qualidade de software especialista em testes unitarios.\n"
        "Seu objetivo e gerar testes para codigo ja convertido.\n"
        "Sempre use a ferramenta generate_unit_tests para montar a resposta final.\n"
        "Nunca faca conversao de codigo, nunca consulte RAG e nunca execute testes.\n"
        "Retorne somente o codigo de teste final."
    )

    if HAS_LANGCHAIN_CREATE_AGENT:
        return _create_agent(
            model=llm,
            tools=[generate_unit_tests],
            system_prompt=system_prompt,
        )

    return _create_react_agent(
        llm,
        [generate_unit_tests],
        prompt=SystemMessage(content=system_prompt),
    )
