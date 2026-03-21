import streamlit as st
import os
from dotenv import load_dotenv
from agent_react import create_agent
from rag.loader import load_and_chunk_pdfs
from rag.embeddings import create_and_store_embeddings
import sys
from io import StringIO
import re

st.set_page_config(page_title="IA Code Conversion (ReAct+Gemini)", layout="wide")

@st.cache_resource
def init_agent():
    return create_agent()

def setup_environment():
    load_dotenv()
    
    # Tratando o custom GEMINI_API_KEY se fornecido no env
    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")
        
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY não encontrada no arquivo .env.")
        st.stop()
        
    # Inicializando o banco de RAG silenciosamente
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        
def parse_agent_output(raw_output: str):
    # Procura blocos de markdown no output final da resposta (para separar o código convertido do teste)
    code_blocks = re.findall(r"```(.*?)\n(.*?)```", raw_output, re.DOTALL)
    return code_blocks

def main():
    setup_environment()
    agent = init_agent()

    st.title("🤖 IA Agent - Code Conversion Tool")
    st.markdown("Converta seu código para qualquer linguagem de destino com testes unitários automáticos e validação, usando o poder do **Agent ReAct** e **Google Gemini**.")

    with st.sidebar:
        st.header("⚙️ Configurações")
        target_lang = st.text_input("Linguagem alvo (ex: javascript, go, python):", "javascript")
        st.markdown("---")
        st.write("📁 **Base de Conhecimento RAG**")
        st.write("Coloque arquivos `.pdf` na pasta `docs/` e clique abaixo para reconstruir o índice vetorial com o novo context da sua empresa.")
        if st.button("Reconstruir Base FAISS"):
            docs_dir = os.path.join(os.path.dirname(__file__), "docs")
            chunks = load_and_chunk_pdfs(docs_dir)
            if chunks:
                create_and_store_embeddings(chunks)
                st.success(f"Base vetorial reconstruída com sucesso ({len(chunks)} trechos).")
            else:
                st.warning("Nenhum PDF encontrado na pasta docs.")

    st.subheader("📝 Código Original")
    source_code = st.text_area("Insira o código aqui:", height=200, placeholder="def sum(a, b):\n    return a + b")

    if st.button("🚀 Iniciar Conversão ReAct", type="primary"):
        if not source_code.strip():
            st.error("Por favor, insira um código válido para converter.")
            return
            
        input_task = f"Converta este código para {target_lang}. Código original: ```\n{source_code}\n```"
        
        with st.status("Processando Ciclo ReAct...", expanded=True) as status:
            # Capturando stdout para exibir logs de pensamento no streamlit
            old_stdout = sys.stdout
            sys.stdout = my_stdout = StringIO()
            
            try:
                response = agent.invoke({"messages": [("user", input_task)]})
                raw_log = my_stdout.getvalue()
                
                status.update(label="Conversão concluída!", state="complete", expanded=False)
                
                # Exibe log ReAct em um expanser
                with st.expander("Ver Logs do ReAct (Thought / Action / Observation)"):
                    st.text(raw_log)
                    
                st.subheader("🎯 Resultado")
                final_answer = response["messages"][-1].content
                
                blocks = parse_agent_output(final_answer)
                
                # Se não conseguiu separar blocos de código via regex, exibe o texto bruto
                if not blocks:
                    st.markdown(final_answer)
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Código Convertido")
                        st.code(blocks[0][1], language=blocks[0][0])
                        
                    if len(blocks) > 1:
                        with col2:
                            st.markdown("### Testes Gerados")
                            st.code(blocks[1][1], language=blocks[1][0])
                            
                    else:
                        st.markdown("### Documentação / Testes Relatados")
                        st.markdown(final_answer)

            except Exception as e:
                status.update(label="Falha na execução do Agente", state="error")
                st.error(f"Erro: {str(e)}")
            finally:
                sys.stdout = old_stdout

if __name__ == "__main__":
    main()
