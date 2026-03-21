import os
from dotenv import load_dotenv

from agent_react import create_agent
from rag.loader import load_and_chunk_pdfs
from rag.embeddings import create_and_store_embeddings

def initialize_rag():
    print("Iniciando carregamento do RAG...")
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"Diretório {docs_dir} criado. Adicione PDFs lá se desejar testar o RAG.")
    
    chunks = load_and_chunk_pdfs(docs_dir)
    if chunks:
        print(f"Foram encontrados {len(chunks)} trechos nos PDFs. Criando banco vetorial...")
        create_and_store_embeddings(chunks)
        print("Banco FAISS criado com sucesso.")
    else:
        print("Continuando sem banco de RAG atualizado.")
        
def main():
    load_dotenv()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("AVISO: A variável de ambiente OPENAI_API_KEY não foi encontrada.")
        print("Por favor, crie um arquivo .env e adicione sua chave lá.")
        return

    # 1. Carregar/Inicializar base de documentos FAISS
    initialize_rag()
    
    # 2. Iniciar Agent
    print("\n" + "="*50)
    print("🤖 IA Agent - Code Conversion Tool (ReAct)")
    print("="*50)
    
    agent = create_agent()
    
    # Exemplo interativo com CLI simples
    while True:
        print("\nPara sair digite 'exit' ou pressione Ctrl+C")
        source_code_input = input("Digite ou cole o código original (em uma linha ou informe um caminho de arquivo, ex: 'def sum(a,b): return a + b'):\n> ")
        if source_code_input.lower() in ("exit", "quit"):
            break
            
        target_lang = input("Linguagem alvo (ex: javascript, go): ")
        
        # Tratamento de input simples para arquivos locais
        if os.path.exists(source_code_input):
            with open(source_code_input, "r", encoding="utf-8") as f:
                source_code = f.read()
        else:
            source_code = source_code_input
            
        input_task = f"Converta este código para {target_lang}. Código original: ```\n{source_code}\n```"
        
        # O agent tem acesso ao histórico se necessário, ou só RAG
        print("\nIniciando Ciclo ReAct...\n")
        try:
            response = agent.invoke({"messages": [("user", input_task)]})
            final_answer = response["messages"][-1].content
            print("\n================ RESULTADO FINAL ================\n")
            print(final_answer)
            print("\n=======================================================\n")
        except Exception as e:
            print(f"\nErro fatal durante execução do Agente: {str(e)}")

if __name__ == "__main__":
    main()
