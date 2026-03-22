from __future__ import annotations

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from providers import get_chat_model


def _normalize_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


@tool
def generate_unit_tests(
    converted_code: str,
    language: str,
    framework: str,
    extra_instructions: str = "",
) -> str:
    """Gera testes unitarios para um codigo ja convertido."""
    print(
        f"[REACT] ACTION | generate_unit_tests | "
        f"language={language}, framework={framework}, chars={len(converted_code)}"
    )

    llm = get_chat_model(temperature=0)

    prompt = PromptTemplate.from_template(
        "Voce e um engenheiro especialista em testes unitarios.\n"
        "Sua tarefa e gerar testes para o codigo fornecido.\n"
        "Regras:\n"
        "1. Retorne apenas codigo de teste, sem explicacoes.\n"
        "2. Use o framework informado em {framework}.\n"
        "3. Cubra casos de sucesso, erro e borda quando fizer sentido.\n"
        "4. Nao altere o codigo de producao.\n"
        "5. Se houver dependencias externas, use mocks.\n\n"
        "Linguagem: {language}\n"
        "Framework de teste: {framework}\n"
        "Instrucoes adicionais: {extra_instructions}\n\n"
        "Codigo ja convertido:\n{converted_code}\n"
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "converted_code": converted_code,
            "language": language,
            "framework": framework,
            "extra_instructions": extra_instructions or "Nenhuma",
        }
    )
    result = _normalize_content_to_text(response.content).strip()
    print(
        f"[REACT] OBSERVATION | generate_unit_tests | "
        f"saida_com_{len(result)}_caracteres"
    )
    return result
