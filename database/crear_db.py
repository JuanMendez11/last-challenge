import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIGURACIÓN ---
NOMBRE_ARCHIVO_CSV = "recetas_arg_limpias.csv"
CARPETA_BASE_DATOS = "./db_recetas_argentinas"

# El modelo liviano y rápido que elegiste
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- FUNCION 1: Cargar y Dividir (Splitting) ---
def cargar_y_dividir_dataframe(df):
    print(f"--- Procesando {len(df)} recetas ---")
    
    docs_finales = []
    
    # Configuración ideal para MiniLM (Chunks de 500-800 chars funcionan perfecto)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # Tamaño del pedazo
        chunk_overlap=150,   # Solapamiento para mantener contexto
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for index, row in df.iterrows():
        # 1. Contenido base
        contenido_completo = f"""
        Nombre: {row['Nombre']}
        Categoría: {row['Categoria']}
        Tiempo de preparación: {row['Duracion']}
        Ingredientes: {row['Ingredientes']}
        Pasos: {row['Pasos']}
        """
        
        # 2. Metadatos base (con ID original)
        metadatos_base = {
            "id_original": str(index),
            "nombre": str(row['Nombre']),
            "categoria": str(row['Categoria']),
            "duracion": str(row['Duracion'])
        }
        
        # 3. Crear fragmentos
        fragmentos = splitter.create_documents(
            texts=[contenido_completo], 
            metadatas=[metadatos_base]
        )
        
        # 4. Enriquecer fragmentos (Contextualización)
        for i, frag in enumerate(fragmentos):
            # Les agregamos info para que no queden huérfanos
            frag.metadata["parte"] = i + 1
            
            # TRUCO: Si es una continuación, repetimos el nombre al inicio
            # para que el buscador sepa de qué receta habla este pedazo suelto.
            if i > 0:
                frag.page_content = f"Receta {row['Nombre']} (Continuación):\n" + frag.page_content
            
            docs_finales.append(frag)
            
    return docs_finales

# --- FUNCION 2: Guardar Vector DB ---
def guardar_vector_db(documentos, path_db):
    print(f"\n--- Iniciando Embeddings Locales ---")
    print(f"Modelo: {MODELO_EMBEDDING}")
    print(f"Total de fragmentos a vectorizar: {len(documentos)}")
    
    # Configuración optimizada para CPU
    embedding_func = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDING,
        encode_kwargs={'batch_size': 32} # Procesa de a 32 para ir rápido pero seguro
    )
    
    print(f"Guardando en '{path_db}'...")
    
    Chroma.from_documents(
        documents=documentos,
        embedding=embedding_func,
        persist_directory=path_db
    )
    
    print(f"✅ ¡ÉXITO! Base de datos creada correctamente.")

# --- MAIN ---
if __name__ == "__main__":
    try:
        # 1. Leer CSV
        print("Leyendo archivo...")
        df = pd.read_csv(NOMBRE_ARCHIVO_CSV)
        df = df.astype(str)
        
        # 2. Convertir y dividir
        mis_docs = cargar_y_dividir_dataframe(df)
        
        # 3. Vectorizar y Guardar
        guardar_vector_db(mis_docs, CARPETA_BASE_DATOS)
        
    except Exception as e:
        print(f"❌ Error: {e}")