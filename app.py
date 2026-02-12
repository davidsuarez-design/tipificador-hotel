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

# --- CARGAR MODELO DE LENGUAJE ---
@st.cache_resource
def cargar_detector_nombres():
    # Carga el modelo instalado desde requirements.txt
    try:
        return spacy.load("es_core_news_sm")
    except:
        return None

nlp = cargar_detector_nombres()

# --- ENCABEZADO ---
col_logo, col_titulo = st.columns([1, 4])
with col_titulo:
    st.title("Sistema de Inteligencia Artificial")
    st.subheader("Tipificación y Detección de Nombres")

# --- FUNCIONES ---
def limpiar_texto_simple(texto):
    if pd.isna(texto): return ""
    return str(texto).lower().strip()

def procesar_separacion_guiones(df, col_comentario):
    """Separa comentarios unidos por guiones '-'"""
    df_exp = df.copy()
    df_exp[col_comentario] = df_exp[col_comentario].astype(str)
    
    # Separar por guion y crear nuevas filas
    df_exp[col_comentario] = df_exp[col_comentario].str.split('-')
    df_exp = df_exp.explode(col_comentario)
    
    # Limpiar
    df_exp[col_comentario] = df_exp[col_comentario].str.strip()
    df_exp = df_exp[df_exp[col_comentario].str.len() > 1] # Eliminar vacíos
    
    df_exp.reset_index(drop=True, inplace=True)
    return df_exp

def verificar_nombres(texto):
    """Detecta nombres de personas (PER)"""
    if nlp is None: return "Error Modelo"
    if pd.isna(texto) or texto == "": return "No validar"
    
    doc = nlp(str(texto))
    for entidad in doc.ents:
        if entidad.label_ == "PER":
            return "Validar"
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
        
        # Buscar columna comentario
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

# --- ENTRENAMIENTO ---
@st.cache_resource
def entrenar_modelos(df_train):
    with st.spinner('Entrenando cerebro digital...'):
        df = df_train.copy()
        df['clean_text'] = df['Comentario'].apply(limpiar_texto_simple)
        
        stop_phrases = ['no', 'no.', 'ninguno', 'ninguna', 'sin comentarios', 'ok', 'na', 'no aplica']
        df = df[~df['clean_text'].isin(stop_phrases)]
        df = df[df['clean_text'].str.len() > 3]

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
    archivo_entrenar = st.file_uploader("1. Sube Histórico", type=["csv", "xlsx"], key="train")
    
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
        st.caption("Precisión:")
        st.progress(st.session_state['metricas']['Area'], text=f"Área: {st.session_state['metricas']['Area']:.0%}")

st.write("Sube el archivo. Se separará por guiones (`-`) y se validarán nombres.")

archivo_predecir = st.file_uploader("2. Sube Nuevas Encuestas", type=["csv", "xlsx"], key="pred")

if archivo_predecir and 'modelos' in st.session_state:
    df_new = cargar_archivo_inteligente(archivo_predecir)
    
    if df_new is not None:
        if st.button("Procesar y Tipificar 🚀"):
            st.info(f"Filas originales: {len(df_new)}")
            
            # 1. Separar
            df_exp = procesar_separacion_guiones(df_new, 'Comentario')
            st.info(f"Filas tras separar guiones: {len(df_exp)}")
            
            # 2. Predecir
            txt = df_exp['Comentario'].apply(limpiar_texto_simple)
            modelos = st.session_state['modelos']
            
            df_exp['Pred_Area'] = modelos['Area'].predict(txt)
            df_exp['Pred_Tipo'] = modelos['Tipo'].predict(txt)
            df_exp['Pred_Sentimiento'] = modelos['Sentimiento'].predict(txt)
            
            # 3. Detectar Nombres
            with st.spinner('Buscando nombres...'):
                df_exp['Validación_Nombre'] = df_exp['Comentario'].apply(verificar_nombres)
            
            # 4. Resultado
            st.dataframe(df_exp.head(10))
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_exp.to_excel(writer, index=False)
            
            st.download_button("Descargar Excel", buffer.getvalue(), "Tipificacion_Final.xlsx", "application/vnd.ms-excel")

elif archivo_predecir and 'modelos' not in st.session_state:
    st.warning("⚠️ Entrena el modelo primero.")

