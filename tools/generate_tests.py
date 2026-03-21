from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

@tool
def generate_tests(converted_code: str, target_lang: str) -> str:
    """
    Gera testes unitários para o código convertido.
    O target_lang deve ser a linguagem do código convertido (ex: python, javascript).
    Retorna apenas o código dos testes.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0)
    
    prompt = PromptTemplate.from_template(
        "Você é um engenheiro de software especialista em testes automatizados.\n"
        "Escreva testes unitários completos e executáveis em {target_lang} para o seguinte código.\n"
        "Use um framework padrão da linguagem (ex: pytest para Python, jest/mocha puro para JavaScript, unittests, etc).\n"
        "Retorne APENAS o código dos testes, garantindo que contenha os imports necessários para que possa ser salvo e executado diretamente.\n"
        "Se o código alvo precisa ser importado, assuma que ele está no mesmo diretório no arquivo 'converted_target.{ext}' "
        "onde ext é a extensão usual (ex: .py para python, .js para js).\n\n"
        "Código a testar:\n{converted_code}"
    )
    
    ext = "js" if target_lang.lower() in ["javascript", "js", "node"] else "py"
    
    chain = prompt | llm
    response = chain.invoke({
        "target_lang": target_lang,
        "converted_code": converted_code,
        "ext": ext
    })
    
    content = response.content
    if content.startswith("```"):
        lines = content.split("\n")
        if len(lines) > 2:
            return "\n".join(lines[1:-1])
    return content
