from langchain_core.tools import tool

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import query_rag_index

@tool
def query_rag(query: str) -> str:
    """
    Consulta a base vetorial interna da empresa carregada de PDFs.
    Use isto ANTES de gerar o código convertido para buscar padrões, 
    guidelines ou regras arquiteturais com base na linguagem ou biblioteca 
    para garantir que a conversão siga os padrões da empresa.
    """
    return query_rag_index(query)
