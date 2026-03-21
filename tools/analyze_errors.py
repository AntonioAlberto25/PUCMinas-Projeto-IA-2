from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

@tool
def analyze_errors(test_result: str) -> str:
    """
    Analisa a saída de um teste que falhou e sugere correções ou modificações
    para o código convertido ou para o código de teste.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0)
    
    prompt = PromptTemplate.from_template(
        "Você é um engenheiro de software debugar.\n"
        "O seguinte relatório de teste tem falhas. Analise o erro e retorne as causas prováveis e como corrigir o código fonte gerado.\n\n"
        "Relatório do Teste:\n{test_result}"
    )
    
    chain = prompt | llm
    response = chain.invoke({"test_result": test_result})
    return response.content
