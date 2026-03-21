import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdfs(docs_dir: str):
    """
    Carrega todos os arquivos PDF do diretório especificado 
    e os divide em chunks menores para armazenamento em vetor.
    """
    if not os.path.exists(docs_dir):
        print(f"Diretório {docs_dir} não encontrado. Nenhum PDF carregado.")
        return []
        
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    if not pdf_files:
        print(f"Nenhum arquivo PDF encontrado em {docs_dir}.")
        return []
        
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks
