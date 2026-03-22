from __future__ import annotations

import re
from io import StringIO
import sys

import streamlit as st

from agent_react import create_agent
from providers import ensure_env_config


st.set_page_config(page_title="Agente ReAct - Testes Unitarios", layout="wide")


@st.cache_resource
def init_agent():
    return create_agent()


def parse_code_blocks(raw_output: str):
    return re.findall(r"```(.*?)\n(.*?)```", raw_output, re.DOTALL)


def normalize_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            else:
                text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part)
    return str(content)


def parse_react_trace(raw_log: str) -> list[tuple[str, str]]:
    steps: list[tuple[str, str]] = []
    instrumented = re.compile(r"\[REACT\]\s+(ACTION|OBSERVATION)\s+\|\s+([^|]+)\|\s*(.*)")

    for line in raw_log.splitlines():
        match = instrumented.search(line)
        if not match:
            continue
        kind = match.group(1).title()
        tool_name = match.group(2).strip()
        detail = match.group(3).strip()
        steps.append((kind, f"{tool_name}: {detail}"))

    return steps


def extract_final_answer(messages, fallback: str = "") -> str:
    for message in reversed(messages):
        text = normalize_content_to_text(getattr(message, "content", "")).strip()
        if text:
            return text
    return fallback


def main() -> None:
    is_ok, message = ensure_env_config()
    if not is_ok:
        st.error(message)
        st.stop()

    agent = init_agent()

    st.title("ReAct - Gerador de Testes Unitarios")
    st.markdown("Cole seu codigo ja convertido e gere testes unitarios automaticamente.")

    with st.sidebar:
        st.header("Configuracao de teste")
        language = st.text_input("Linguagem do codigo", value="python")
        framework = st.text_input("Framework de teste", value="pytest")
        extra_instructions = st.text_area(
            "Instrucoes adicionais",
            value="Priorize casos de borda e uso de mocks quando houver integracoes externas.",
            height=120,
        )

    converted_code = st.text_area(
        "Codigo ja convertido",
        height=320,
        placeholder="Cole aqui o codigo de producao para gerar os testes unitarios.",
    )

    if st.button("Gerar testes unitarios", type="primary"):
        if not converted_code.strip():
            st.error("Cole um codigo valido antes de gerar os testes.")
            return

        user_task = (
            "Gere testes unitarios usando a ferramenta generate_unit_tests com os seguintes parametros:\n"
            f"language={language}\n"
            f"framework={framework}\n"
            f"extra_instructions={extra_instructions}\n"
            f"converted_code=```\n{converted_code}\n```"
        )

        with st.status("Executando agente ReAct...", expanded=False):
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                response = agent.invoke({"messages": [("user", user_task)]})
                execution_log = sys.stdout.getvalue()
            except Exception as exc:
                error_text = str(exc)
                if "not a valid model id" in error_text.lower() or "invalid model" in error_text.lower():
                    st.error(
                        "Falha na geracao dos testes: modelo invalido no .env. "
                        "Defina LLM_MODEL com um ID valido do seu provedor. "
                        "Exemplo OpenRouter: meta-llama/llama-3.1-8b-instruct:free"
                    )
                elif "no endpoints found" in error_text.lower() or "error code: 404" in error_text.lower():
                    st.error(
                        "Falha na geracao dos testes: o modelo escolhido nao tem endpoint disponivel agora no OpenRouter. "
                        "Use LLM_MODEL=openrouter/auto no .env para fallback automatico."
                    )
                else:
                    st.error(f"Falha na geracao dos testes: {error_text}")
                return
            finally:
                sys.stdout = old_stdout

        trace_steps = parse_react_trace(execution_log)
        if trace_steps:
            with st.expander("Rastro ReAct (Action / Observation)", expanded=False):
                for idx, (kind, content) in enumerate(trace_steps, start=1):
                    st.markdown(f"**{idx}. {kind}**")
                    st.write(content)
        else:
            st.info("Sem rastros instrumentados nesta execucao.")

        with st.expander("Log bruto da execucao", expanded=False):
            st.text(execution_log or "Sem logs disponiveis.")

        final_answer = extract_final_answer(response.get("messages", []))
        if not final_answer:
            st.error("O agente nao retornou conteudo textual. Tente novamente com um trecho menor de codigo.")
            return

        code_blocks = parse_code_blocks(final_answer)

        st.subheader("Resultado")
        if code_blocks:
            block_language = code_blocks[0][0].strip() or language
            block_content = code_blocks[0][1]
            st.code(block_content, language=block_language)
        else:
            st.code(final_answer, language=language)


if __name__ == "__main__":
    main()
