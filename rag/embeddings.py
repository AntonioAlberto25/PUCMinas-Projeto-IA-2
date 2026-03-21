import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")

def create_and_store_embeddings(chunks: list[Document]) -> FAISS:
    """
    Gera embeddings para os chunks de texto usando a API do Google
    e os armazena usando FAISS. Salva o banco vetorial no disco.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    return vectorstore

def load_vectorstore() -> FAISS | None:
    """
    Carrega o banco vetorial FAISS salvo no disco.
    """
    if not os.path.exists(VECTOR_DB_PATH):
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True # Permitido pois os arquivos são estritamente locais e controlados
    )
    return vectorstore
