from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
# Import tools
from tools.convert_code import convert_code
from tools.generate_tests import generate_tests
from tools.run_tests import run_tests
from tools.analyze_errors import analyze_errors
from tools.query_rag import query_rag

def create_agent():
    """
    Cria e configura o Agente ReAct com as ferramentas e o system prompt.
    Usa langgraph.prebuilt.create_react_agent com o parâmetro `prompt`
    (substitui o deprecated `state_modifier`).
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0)
    
    tools = [
        query_rag,
        convert_code,
        generate_tests,
        run_tests,
        analyze_errors
    ]
    
    system_message = SystemMessage(content=
        "Você é um Engenheiro de Software Sênior especializado em IA Generativa, refatoração e migração de código.\n"
        "Seu objetivo é converter o código fonte fornecido para a linguagem de destino, gerar testes unitários e validá-los.\n"
        "Siga rigorosamente estas REGRAS COGNITIVAS:\n"
        "1. Você NÃO PODE gerar o código final diretamente. DEVER usar a ferramenta `query_rag` PRIMEIRO.\n"
        "2. Em seguida, converta o código usando a ferramenta `convert_code`.\n"
        "3. Após converter, use `generate_tests` para gerar testes unitários na linguagem de destino.\n"
        "4. Execute os testes usando a ferramenta `run_tests`.\n"
        "5. Se falharem, use `analyze_errors` e TENTE NOVAMENTE corrigindo o código ou o teste.\n"
        "6. Se os testes passarem, escreva a resposta final com o código e os testes formatados em blocos Markdown."
    )

    # O parâmetro correto nas versões recentes do langgraph é `prompt`
    agent_executor = create_react_agent(llm, tools, prompt=system_message)
    
    return agent_executor
