from langchain_core.tools import tool
import sys
import os

# Adds the agent root to the path so we can import validator when running the agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator.validator import execute_tests

@tool
def run_tests(test_code: str, source_code: str, target_lang: str) -> str:
    """
    Executa os testes gerados contra o código convertido.
    Para chamar essa tool, forneça o `test_code`, o `source_code` (o código convertido)
    e a linguagem alvo `target_lang` (ex: python, javascript).
    Retorna o log bruto de execução detalhando se passou ou falhou.
    """
    result = execute_tests(source_code, test_code, target_lang)
    output_prefix = "TESTES APROVADOS:\n" if result["success"] else "TESTES REPROVADOS:\n"
    return output_prefix + result["output"]
