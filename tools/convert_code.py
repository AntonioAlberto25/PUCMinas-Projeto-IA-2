import os
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

@tool
def convert_code(source_code: str, source_lang: str, target_lang: str, context: str = "") -> str:
    """
    Converte o código fonte (source_code) da linguagem de origem (source_lang) 
    para a linguagem de destino (target_lang). 
    Pode usar um texto de contexto arquitetural adicional (context) se fornecido.
    Retorna apenas o código convertido.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0)
    
    prompt = PromptTemplate.from_template(
        "Você é um engenheiro de software especialista.\n"
        "Sua tarefa é converter o seguinte código escrito em {source_lang} para {target_lang}.\n"
        "{context_section}\n"
        "Retorne APENAS o código convertido, sem explicações extras, preferencialmente dentro de blocos de crase Markdown.\n\n"
        "Código Fonte:\n{source_code}"
    )
    
    context_section = f"Siga as seguintes diretrizes arquiteturais da empresa rigorosamente:\n{context}\n" if context else "Siga as melhores práticas da linguagem de destino."
    
    chain = prompt | llm
    response = chain.invoke({
        "source_lang": source_lang,
        "target_lang": target_lang,
        "context_section": context_section,
        "source_code": source_code
    })
    
    # Extrair apenas o bloco de código
    content = response.content
    if content.startswith("```"):
        lines = content.split("\n")
        if len(lines) > 2:
            return "\n".join(lines[1:-1])
    return content
