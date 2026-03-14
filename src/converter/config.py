from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_api_url: str
    llm_api_key: str
    llm_model: str
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_api_url=os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "8000")),
        )
