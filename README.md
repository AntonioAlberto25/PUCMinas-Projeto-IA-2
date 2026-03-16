# PUCMinas-Projeto-IA

Conversor de trechos de codigo orientado por IA, com pipeline em camadas.

## Arquitetura implementada

1. Camada de entrada
- Recebe codigo bruto + linguagem alvo
- Aceita linguagem de origem opcional e instrucoes de estilo

2. Parser + Normalizer
- Detecta linguagem de origem por heuristica
- Normaliza quebras de linha e espacos finais
- Faz chunking por classes/funcoes (AST para Python, regex para linguagens gerais)

3. LLM Core
- Conversao chunk por chunk
- Contexto acumulado entre chunks
- Prompt de sistema com regras de conversao e estilo
- Suporte a endpoint OpenAI-compatible
- Suporte a RAG com PDFs de padroes arquiteturais da empresa

4. Post-processor + Validator
- Reune chunks convertidos
- Remove imports duplicados
- Syntax check para alvo Python
- Gera diff unificado opcional e relatorio de warnings

5. Saida
- Codigo convertido
- Diff opcional
- Relatorio de validacao + chunks convertidos

## Estrutura

```text
.
|-- .env.example
|-- requirements.txt
|-- src/
|   `-- converter/
|       |-- __init__.py
|       |-- api.py
|       |-- config.py
|       |-- llm.py
|       |-- main.py
|       |-- pipeline.py
|       `-- schemas.py
`-- README.md
```

## Como executar

1. Crie e ative um ambiente virtual Python (recomendado)
2. Instale dependencias:

```bash
pip install -r requirements.txt
```

3. Configure variaveis de ambiente:
- Copie `.env.example` para `.env`
- Preencha `LLM_API_KEY` para habilitar conversao
- Sem chave valida, a API bloqueia `POST /convert` com erro `503`

4. Inicie a API:

```bash
uvicorn converter.api:app --app-dir src --reload
```

## Endpoints

### Health

```http
GET /health
```

Resposta:

```json
{
	"status": "ok",
	"mode": "llm"
}
```

### Conversao

```http
POST /convert
Content-Type: application/json
```

Payload exemplo:

```json
{
	"raw_code": "def soma(a, b):\n    return a + b",
	"target_language": "javascript",
	"source_language": "python",
	"company_name": "MinhaEmpresa",
	"rag_enabled": true,
	"style": {
		"language": "pt-br",
		"naming": "camelCase",
		"patterns": ["clean code", "early return"]
	},
	"include_diff": true
}
```

Resposta (resumo):

```json
{
	"source_language": "python",
	"target_language": "javascript",
	"converted_code": "function soma(a, b) { return a + b; }",
	"diff": "--- original ...",
	"report": {
		"syntax_ok": true,
		"warnings": []
	},
	"rag_context_used": true,
	"rag_sources": ["guia-arquitetura.pdf", "padrao-codigo.pdf"],
	"chunks": [
		{
			"id": "chunk-1",
			"original": "...",
			"converted": "..."
		}
	],
	"mode": "llm"
}
```

### Upload de PDFs para RAG

```http
POST /rag/upload-pdfs
Content-Type: multipart/form-data
```

Campo esperado: `files` (um ou mais arquivos `.pdf`).

Resposta exemplo:

```json
{
	"uploaded": 2,
	"chunks_added": 34,
	"files": ["guia-arquitetura.pdf", "padrao-codigo.pdf"]
}
```

### Status do indice RAG

```http
GET /rag/status
```

Resposta exemplo:

```json
{
	"status": "ok",
	"documents": 2,
	"chunks": 34
}
```

## Fluxo RAG da empresa

1. Envie PDFs da empresa pela interface (seção Base RAG)
2. Informe o nome da empresa no campo Empresa/cliente
3. Durante a conversao, a API recupera trechos relevantes dos PDFs
4. O contexto recuperado e enviado ao LLM para seguir padroes arquiteturais
5. A resposta traz `rag_context_used` e `rag_sources` com as fontes aplicadas

## Regras de seguranca de conversao

1. A conversao so roda com LLM configurado corretamente (sem fallback mock)
2. Chave invalida/sem permissao retorna erro de autenticacao (`401`)
3. Se a resposta do modelo nao transformar de fato o trecho, a API rejeita com `422`

## Proximos incrementos recomendados

1. Integrar parser real para linguagens adicionais (Tree-sitter, Babel, JavaParser)
2. Adicionar validadores por linguagem alvo (ESLint, TypeScript, javac)
3. Incluir testes automatizados para cada etapa do pipeline
4. Criar persistencia de historico de conversoes e auditoria
