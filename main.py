from __future__ import annotations

from agent_react import create_agent
from providers import ensure_env_config


def main() -> None:
    is_ok, message = ensure_env_config()
    if not is_ok:
        print(f"Configuracao invalida: {message}")
        return

    agent = create_agent()

    print("\n" + "=" * 60)
    print("Agente ReAct - Geracao de Testes Unitarios")
    print("=" * 60)

    while True:
        print("\nDigite 'exit' para sair.")
        code_input = input("Cole o codigo ja convertido ou caminho de arquivo:\n> ").strip()
        if code_input.lower() in {"exit", "quit"}:
            break

        language = input("Linguagem do codigo (ex: python, java, javascript): ").strip() or "python"
        framework = input("Framework de teste (ex: pytest, jest, junit): ").strip() or "pytest"
        instructions = input("Instrucoes adicionais (opcional): ").strip()

        try:
            try:
                with open(code_input, "r", encoding="utf-8") as file_handle:
                    converted_code = file_handle.read()
            except OSError:
                converted_code = code_input

            task = (
                "Use a ferramenta generate_unit_tests para criar os testes com os dados abaixo:\n"
                f"language={language}\n"
                f"framework={framework}\n"
                f"extra_instructions={instructions or 'Nenhuma'}\n"
                f"converted_code=```\n{converted_code}\n```"
            )

            response = agent.invoke({"messages": [("user", task)]})
            print("\n--- TESTES GERADOS ---\n")
            print(response["messages"][-1].content)
            print("\n----------------------\n")
        except Exception as exc:
            print(f"Falha ao gerar testes: {exc}")


if __name__ == "__main__":
    main()
