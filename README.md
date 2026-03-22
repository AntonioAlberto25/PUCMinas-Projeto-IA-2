# ReAct Unit Test Generator

Projeto reestruturado para um unico objetivo:
- Receber codigo ja convertido
- Gerar testes unitarios com um agente ReAct

Tudo que era conversao de codigo e RAG foi removido.

## O que ficou

- `app.py`: interface Streamlit
- `main.py`: CLI opcional
- `agent_react.py`: agente ReAct
- `tools/generate_unit_tests.py`: ferramenta de geracao de testes
- `providers.py`: configuracao do LLM por `.env`

## Configuracao minima (.env)

Use apenas estas variaveis:

```env
LLM_URL=https://api.openai.com/v1
LLM_KEY=sua_chave
LLM_MODEL=gpt-4o-mini
```

Existe um exemplo em `.env.example`.

## Execucao

### Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

### CLI

```bash
python main.py
```

## Fluxo

1. Cole o codigo ja convertido
2. Informe linguagem e framework de teste
3. O agente ReAct chama a ferramenta `generate_unit_tests`
4. Receba o codigo de testes unitarios
