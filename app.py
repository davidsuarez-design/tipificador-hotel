import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io
import spacy

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Tipificador IA Hotelero", layout="wide")

# --- CARGAR MODELO DE LENGUAJE ESPAÑOL (PARA DETECTAR NOMBRES) ---
@st.cache_resource
def cargar_detector_nombres():
    try:
        # Intenta cargar el modelo de español
        return spacy.load("es_core_news_sm")
    except OSError:
        # Si no está instalado, lo descarga automáticamente
        from spacy.cli import download
        download("es_core_news_sm")
        return spacy.load("es_core_news_sm")

nlp = cargar_detector_nombres()

# --- ENCABEZADO ---
col_logo, col_titulo = st.columns([1, 4])
with col_titulo:
    st.title("Sistema de Inteligencia Artificial")
    st.subheader("Tipificación y Detección de Nombres (Separado por '-')")

# --- FUNCIONES DE LIMPIEZA Y PROCESAMIENTO ---
def limpiar_texto_simple(texto):
    if pd.isna(texto): return ""
    return str(texto).lower().strip()

def procesar_separacion_guiones(df, col_comentario):
    df_exp = df.copy()
    df_exp[col_comentario] = df_exp[col_comentario].astype(str)
    
    # 1. Separar por guiones
    df_exp[col_comentario] = df_exp[col_comentario].str.split('-')
    
    # 2. Explotar (Crear filas nuevas)
    df_exp = df_exp.explode(col_comentario)
    
    # 3. Limpieza de fragmentos
    df_exp[col_comentario] = df_exp[col_comentario].str.strip()
    df_exp = df_exp[df_exp[col_comentario].str.len() > 1] # Ignorar vacíos
    
    df_exp.reset_index(drop=True, inplace=True)
    return df_exp

def verificar_nombres(texto):
    """Analiza el texto y busca nombres de personas (PER)"""
    if pd.isna(texto) or texto == "":
        return "No validar"
    
    # Procesar el texto respetando las mayúsculas originales (importante para detectar nombres)
    doc = nlp(str(texto))
    
    # Buscar si alguna entidad detectada es una Persona ('PER')
    for entidad in doc.ents:
        if entidad.label_ == "PER":
            return "Validar" # Encontró un nombre de persona
            
    return "No validar"

# --- CARGA INTELIGENTE ---
def cargar_archivo_inteligente(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
                if len(df.columns) < 2: raise ValueError()
            except:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
        else:
            df = pd.read_excel(uploaded_file)
            
        df.columns = df.columns.str.strip()
        
        # Normalizar nombre de columna Comentario
        if 'Comentario' not in df.columns:
            posibles = [c for c in df.columns if 'coment' in c.lower() or 'review' in c.lower()]
            if posibles:
                df.rename(columns={posibles[0]: 'Comentario'}, inplace=True)
            else:
                st.error("❌ No encontré la columna 'Comentario'.")
                return None
        return df
    except Exception as e:
        st.error(f"Error leyendo archivo: {e}")
        return None

# --- ENTRENAMIENTO (SIN NPS) ---
@st.cache_resource
def entrenar_modelos(df_train):
    with st.spinner('Entrenando cerebro digital...'):
        df = df_train.copy()
        df['clean_text'] = df['Comentario'].apply(limpiar_texto_simple)
        
        stop_phrases = ['no', 'no.', 'ninguno', 'ninguna', 'sin comentarios', 'ok', 'na', 'no aplica']
        df = df[~df['clean_text'].isin(stop_phrases)]
        df = df[df['clean_text'].str.len() > 3]

        # Solo dejamos Area, Tipo y Sentimiento
        targets = {
            'Area': df['Area'], 
            'Tipo': df['Tipo'], 
            'Sentimiento': df['Clasificación']
        }
        
        modelos = {}
        metricas = {}

        for nombre, y in targets.items():
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))), 
                ('clf', LinearSVC(class_weight='balanced', random_state=42, max_iter=1000))
            ])
            
            X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            acc = accuracy_score(y_test, pipeline.predict(X_test))
            
            modelos[nombre] = pipeline
            metricas[nombre] = acc
            
        return modelos, metricas

# --- INTERFAZ ---

with st.sidebar:
    st.header("⚙️ Configuración")
    archivo_entrenar = st.file_uploader("1. Sube Histórico (Entrenamiento)", type=["csv", "xlsx"], key="train")
    
    if archivo_entrenar:
        df_train = cargar_archivo_inteligente(archivo_entrenar)
        if df_train is not None:
            if st.button("Entrenar Modelo 🧠"):
                modelos, metricas = entrenar_modelos(df_train)
                st.session_state['modelos'] = modelos
                st.session_state['metricas'] = metricas
                st.success("¡Modelo Entrenado!")

    if 'metricas' in st.session_state:
        st.divider()
        st.caption("Precisión del Modelo:")
        st.progress(st.session_state['metricas']['Area'], text=f"Áreas: {st.session_state['metricas']['Area']:.0%}")
        st.progress(st.session_state['metricas']['Sentimiento'], text=f"Sentimiento: {st.session_state['metricas']['Sentimiento']:.0%}")

st.write("Sube el archivo de encuestas. Si un comentario tiene guiones (`-`), se separará en varias filas.")

archivo_predecir = st.file_uploader("2. Sube Nuevas Encuestas", type=["csv", "xlsx"], key="pred")

if archivo_predecir and 'modelos' in st.session_state:
    df_new = cargar_archivo_inteligente(archivo_predecir)
    
    if df_new is not None:
        if st.button("Procesar y Tipificar 🚀"):
            st.info(f"Filas originales: {len(df_new)}")
            
            # 1. Separar por guiones
            df_expandido = procesar_separacion_guiones(df_new, 'Comentario')
            st.info(f"Filas después de separar por guiones (-): {len(df_expandido)}")
            
            # 2. Textos limpios para el modelo predictivo
            textos_limpios = df_expandido['Comentario'].apply(limpiar_texto_simple)
            
            # 3. Predecir Tipificación
            modelos = st.session_state['modelos']
            df_expandido['Pred_Area'] = modelos['Area'].predict(textos_limpios)
            df_expandido['Pred_Tipo'] = modelos['Tipo'].predict(textos_limpios)
            df_expandido['Pred_Sentimiento'] = modelos['Sentimiento'].predict(textos_limpios)
            
            # 4. Detectar Nombres de Personas (Usamos el texto original respetando mayúsculas)
            with st.spinner('Analizando y buscando nombres de personas...'):
                df_expandido['Validación_Nombre'] = df_expandido['Comentario'].apply(verificar_nombres)
            
            # 5. Mostrar y Descargar
            st.dataframe(df_expandido[['Comentario', 'Pred_Area', 'Pred_Tipo', 'Pred_Sentimiento', 'Validación_Nombre']].head(10))
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_expandido.to_excel(writer, index=False)
            
            st.download_button(
                label="Descargar Excel Final", 
                data=buffer.getvalue(), 
                file_name="Tipificacion_Final.xlsx",
                mime="application/vnd.ms-excel"
            )

elif archivo_predecir and 'modelos' not in st.session_state:
    st.warning("⚠️ Recuerda entrenar el modelo primero en el menú de la izquierda.")
