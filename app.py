import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Tipificador IA de Encuestas", layout="wide")

st.title("🤖 Tipificador Automático de Encuestas")
st.markdown("""
Esta aplicación utiliza Inteligencia Artificial para clasificar comentarios de encuestas.
**Pasos:**
1. Carga tu **Histórico** (para que la IA aprenda).
2. Carga tus **Nuevos Comentarios** (para tipificar).
""")

# --- FUNCIONES DE LIMPIEZA ---
def limpiar_texto(df, col_comentario):
    # Asegurar que sea string
    df['clean_text'] = df[col_comentario].astype(str).str.lower().str.strip()
    
    # Lista de comentarios "basura"
    stop_phrases = ['no', 'no.', 'ninguno', 'ninguna', 'sin comentarios', 'ok', 'na', 'no aplica', 'todo bien', 'gracias']
    
    # Filtros
    df_clean = df[~df['clean_text'].isin(stop_phrases)].copy()
    df_clean = df_clean[df_clean['clean_text'].str.len() > 3]
    
    return df_clean

# --- CARGA DE ARCHIVOS INTELIGENTE (Detecta separadores y nombres) ---
def cargar_archivo_inteligente(uploaded_file):
    try:
        # 1. Leer según extensión
        if uploaded_file.name.endswith('.csv'):
            try:
                # Intento A: Separador punto y coma (;), UTF-8
                df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
                if len(df.columns) < 2: # Si falló la separación
                    raise ValueError("Probando otro separador")
            except:
                try:
                    # Intento B: Separador coma (,), UTF-8
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
                except:
                    # Intento C: Latin-1 (Excel viejo)
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
        else:
            # Excel (.xlsx)
            df = pd.read_excel(uploaded_file)
            
        # 2. Normalizar Columnas (Quitar espacios extra: " Area " -> "Area")
        df.columns = df.columns.str.strip()
        
        # 3. Buscar la columna 'Comentario' aunque esté mal escrita
        # Si no existe 'Comentario', buscamos variantes comunes
        if 'Comentario' not in df.columns:
            posibles_nombres = [c for c in df.columns if 'comentario' in c.lower() or 'review' in c.lower()]
            if posibles_nombres:
                st.warning(f"⚠️ No encontré la columna 'Comentario', pero usaré '{posibles_nombres[0]}' que se le parece.")
                df.rename(columns={posibles_nombres[0]: 'Comentario'}, inplace=True)
            else:
                st.error(f"❌ Error: Tu archivo no tiene una columna llamada 'Comentario'. Columnas encontradas: {list(df.columns)}")
                return None
                
        return df

    except Exception as e:
        st.error(f"Error crítico leyendo el archivo: {e}")
        return None

# --- ENTRENAMIENTO ---
@st.cache_resource
def entrenar_modelos(df_train):
    with st.spinner('Entrenando cerebro digital...'):
        # Validar columnas necesarias
        required_cols = ['Area', 'Tipo', 'Clasificación', 'Clasificación NPS']
        missing = [c for c in required_cols if c not in df_train.columns]
        if missing:
            st.error(f"❌ El archivo histórico debe tener las columnas: {missing}")
            return None, None

        df = limpiar_texto(df_train, 'Comentario')
        
        # Variables Objetivo
        targets = {
            'Area': df['Area'],
            'Tipo': df['Tipo'],
            'Sentimiento': df['Clasificación'],
            'NPS': df['Clasificación NPS']
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

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Entrenamiento")
    archivo_entrenamiento = st.file_uploader("Sube tu CSV Histórico", type=["csv", "xlsx"], key="train")

modelos = None

if archivo_entrenamiento:
    df_train = cargar_archivo_inteligente(archivo_entrenamiento)
    
    if df_train is not None:
        st.success(f"Cargados {len(df_train)} registros. Columnas: {list(df_train.columns)}")
        
        if st.button("Entrenar Modelos"):
            modelos, metricas = entrenar_modelos(df_train)
            if modelos:
                st.session_state['modelos'] = modelos
                st.session_state['metricas'] = metricas

# Mostrar métricas
if 'modelos' in st.session_state:
    modelos = st.session_state['modelos']
    m = st.session_state['metricas']
    st.info(f"✅ Modelos Listos. Precisión: NPS ({m['NPS']:.0%}), Área ({m['Area']:.0%})")

with col2:
    st.header("2. Predicción")
    archivo_nuevos = st.file_uploader("Sube nuevas encuestas", type=["csv", "xlsx"], key="predict")
    
    if archivo_nuevos and modelos:
        df_new = cargar_archivo_inteligente(archivo_nuevos)
        
        if df_new is not None:
            if st.button("Tipificar Ahora"):
                df_proc = df_new.copy()
                clean_txt = df_proc['Comentario'].astype(str).str.lower().str.strip()
                
                df_proc['Pred_Area'] = modelos['Area'].predict(clean_txt)
                df_proc['Pred_Tipo'] = modelos['Tipo'].predict(clean_txt)
                df_proc['Pred_Sentimiento'] = modelos['Sentimiento'].predict(clean_txt)
                df_proc['Pred_NPS'] = modelos['NPS'].predict(clean_txt)
                
                st.write("Vista previa:")
                st.dataframe(df_proc[['Comentario', 'Pred_Area', 'Pred_Tipo', 'Pred_NPS']].head())
                
                # Descarga
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_proc.to_excel(writer, index=False)
                    
                st.download_button("Descargar Excel", buffer.getvalue(), "Encuestas_Tipificadas.xlsx", "application/vnd.ms-excel")
    
    elif archivo_nuevos and not modelos:
        st.warning("⚠️ Primero entrena los modelos en el paso 1.")


