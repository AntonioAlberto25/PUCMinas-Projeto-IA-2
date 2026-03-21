# Agente de Conversão de Código (ReAct + Google Gemini + Streamlit)

Um agente de inteligência artificial autônomo baseado na arquitetura ReAct (Reasoning and Acting) para converter código de uma linguagem para outra, gerar testes e os validar automaticamente. 
Conta com suporte a RAG (Retrieval-Augmented Generation) para garantir que o código siga as diretrizes arquiteturais e padrões da empresa.

## Estrutura do Projeto

```text
.
|-- app.py                 # Interface Web com Streamlit
|-- main.py                # Opção de execução CLI Interativo
|-- setup_and_run.ps1      # Script automatizado (criação da venv e execução)
|-- agent_react.py         # Configuração do Agente Langchain ReAct com Gemini
|-- tools/                 # Ferramentas usadas pelo agente nas actions
|   |-- convert_code.py    # Conversão via Gemini
|   |-- generate_tests.py  # Geração de testes unitários via Gemini
|   |-- run_tests.py       # Wrapper para o Validator
|   |-- analyze_errors.py  # Análise de erros de compilação via Gemini
|   `-- query_rag.py       # Busca vetorial de padrões da empresa
|-- rag/                   # Módulo Vetorial
|   |-- loader.py          # Leitura de PDFs
|   |-- embeddings.py      # Vetorização (Gemini Embeddings) e FAISS
|   `-- retriever.py       # Busca semântica
|-- validator/             # Execução isolada do código gerado
|   `-- validator.py       # Executa testes gerados sob Node / Pytest e capta logs
`-- docs/                  # Pasta para manuais PDF com arquitetura da empresa
```

## Requisitos
* Python 3.10 ou superior
* `node` (NodeJS) instalado globalmente (para testar conversões JS).
* Conta Google com API Key do Gemini (AI Studio).

## Instalação Rápida e Execução (Automática)
> **Recomendado para Windows:** Utilizamos um script PowerShell que faz tudo por você!

1. Preencha sua chave de API
Copie ou crie arquivo o `.env` baseado no `.env.example`:
```text
GEMINI_API_KEY=sua_chave_gerada_no_google_ai_studio
```

2. Execute o script de Setup:
Basta clicar com o botão direito no arquivo `setup_and_run.ps1` e escolher **Executar com o PowerShell** (ou executá-lo pelo terminal). Ele irá:
- Criar a pasta isolada `venv`.
- Ativar o ambiente automático.
- Instalar bibliotecas de `requirements.txt`.
- Inicializar a interface gráfica web `app.py` no seu navegador usando o Streamlit.

## Instalação e Execução Manual
Se preferir não usar o script `setup_and_run.ps1`, faça manualmente no seu terminal local:

1. Crie a venv e ative-a:
```bash
python -m venv venv
# No windows:
.\venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Inicie o Streamlit:
```bash
streamlit run app.py
```

## Configuração do RAG (Base de Conhecimento)
- Coloque os PDFs contendo os padrões da sua empresa dentro da pasta `docs/`.
- Acesse a interface web do aplicativo Streamlit e clique no botão lateral **"Reconstruir Base FAISS"**. O LangChain lerá os PDFs recém-adicionados e os integrará no escopo de conhecimento do Agente durante todas as próximas conversões.
