from .embeddings import load_vectorstore

def query_rag_index(query: str, k: int = 3) -> str:
    """
    Realiza uma busca semântica na base de conhecimento (PDFs internos) 
    para retornar os documentos mais relevantes com contexto arquitetural.
    
    Retorna uma string contendo o conteúdo combinado dos documentos recuperados,
    ou uma mensagem informando que nada relevante foi encontrado.
    """
    vectorstore = load_vectorstore()
    
    if vectorstore is None:
        return "Nenhuma base de conhecimento (RAG) encontrada. É necessário carregar os PDFs primeiro."
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    
    if not docs:
        return "Nenhum documento relevante encontrado para a query."
        
    context = "\n\n---\n\n".join(
        [f"Fonte: {doc.metadata.get('source', 'Desconhecida')} (Página {doc.metadata.get('page', 'N/A')})\nConteúdo:\n{doc.page_content}" 
         for doc in docs]
    )
    return context
