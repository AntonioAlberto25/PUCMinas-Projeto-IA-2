from __future__ import annotations

import os

from dotenv import load_dotenv


def ensure_env_config() -> tuple[bool, str]:
    """Valida as variaveis minimas necessarias para uso do LLM."""
    load_dotenv()

    llm_url = (os.environ.get("LLM_URL") or "").strip()
    llm_key = (os.environ.get("LLM_KEY") or "").strip()
    llm_model = (os.environ.get("LLM_MODEL") or "").strip()

    if not llm_url:
        return False, "Defina LLM_URL no arquivo .env."
    if not llm_key:
        return False, "Defina LLM_KEY no arquivo .env."
    if not llm_model:
        return False, "Defina LLM_MODEL no arquivo .env."
    return True, "ok"


def get_chat_model(temperature: float = 0):
    """Cria um cliente de chat OpenAI-compatible com base nas variaveis do .env."""
    load_dotenv()

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=(os.environ.get("LLM_MODEL") or "").strip(),
        api_key=(os.environ.get("LLM_KEY") or "").strip(),
        base_url=(os.environ.get("LLM_URL") or "").strip(),
        temperature=temperature,
    )
