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
    # Convertir a string y minúsculas
    df['clean_text'] = df[col_comentario].astype(str).str.lower().str.strip()
    
    # Lista de comentarios "basura" que no aportan valor
    stop_phrases = ['no', 'no.', 'ninguno', 'ninguna', 'sin comentarios', 'ok', 'na', 'no aplica', 'todo bien', 'gracias']
    
    # Filtro 1: Eliminar frases exactas de la lista basura
    df_clean = df[~df['clean_text'].isin(stop_phrases)].copy()
    
    # Filtro 2: Eliminar comentarios muy cortos (menos de 4 letras)
    df_clean = df_clean[df_clean['clean_text'].str.len() > 3]
    
    return df_clean

# --- FUNCIÓN AUXILIAR PARA LEER ARCHIVOS (CORRECCIÓN UTF-8/LATIN) ---
def cargar_archivo_seguro(uploaded_file):
    """Intenta leer el archivo con diferentes codificaciones para evitar errores."""
    if uploaded_file.name.endswith('.csv'):
        try:
            # Intento 1: UTF-8 (Estándar moderno)
            return pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # Intento 2: Latin-1 (Excel en Español/Windows)
                return pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
            except:
                st.error("No se pudo leer el CSV. Verifica que esté separado por punto y coma (;)")
                return None
    else:
        return pd.read_excel(uploaded_file)

# --- ENTRENAMIENTO DEL MODELO ---
@st.cache_resource
def entrenar_modelos(df_train):
    with st.spinner('Entrenando cerebro digital... esto puede tomar unos segundos.'):
        # Limpieza inicial
        df = limpiar_texto(df_train, 'Comentario')
        
        # Variables Objetivo
        X = df['clean_text']
        targets = {
            'Area': df['Area'],
            'Tipo': df['Tipo'],
            'Sentimiento': df['Clasificación'],
            'NPS': df['Clasificación NPS']
        }
        
        modelos = {}
        metricas = {}

        # Entrenamos un modelo independiente para cada columna objetivo
        for nombre, y in targets.items():
            # Pipeline: Vectorización + Clasificador
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))), 
                ('clf', LinearSVC(class_weight='balanced', random_state=42, max_iter=1000))
            ])
            
            # Dividir para validar
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            pipeline.fit(X_train, y_train)
            acc = accuracy_score(y_test, pipeline.predict(X_test))
            
            modelos[nombre] = pipeline
            metricas[nombre] = acc
            
        return modelos, metricas

# --- INTERFAZ DE USUARIO ---

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Entrenamiento")
    archivo_entrenamiento = st.file_uploader("Sube tu CSV Histórico", type=["csv", "xlsx"])

modelos = None

if archivo_entrenamiento:
    try:
        df_train = cargar_archivo_seguro(archivo_entrenamiento)
            
        if df_train is not None:
            st.success(f"Cargados {len(df_train)} registros históricos.")
            
            # Botón para iniciar entrenamiento
            if st.button("Entrenar Modelos"):
                modelos, metricas = entrenar_modelos(df_train)
                st.session_state['modelos'] = modelos # Guardar en sesión
                st.session_state['metricas'] = metricas

    except Exception as e:
        st.error(f"Error inesperado: {e}")

# Mostrar métricas si ya existen
if 'modelos' in st.session_state:
    modelos = st.session_state['modelos']
    st.info(f"✅ Modelos Listos. Precisión estimada: NPS ({st.session_state['metricas']['NPS']:.0%}), Área ({st.session_state['metricas']['Area']:.0%})")

with col2:
    st.header("2. Predicción (Nuevos Datos)")
    archivo_nuevos = st.file_uploader("Sube el archivo SIN tipificar", type=["csv", "xlsx"])
    
    if archivo_nuevos and modelos:
        df_new = cargar_archivo_seguro(archivo_nuevos)
            
        if df_new is not None:
            if 'Comentario' not in df_new.columns:
                st.error("⚠️ El archivo debe tener una columna llamada 'Comentario' (escrito exactamente así).")
            else:
                # Predecir
                if st.button("Tipificar Ahora"):
                    df_procesado = df_new.copy()
                    
                    # Limpiamos solo para la predicción
                    textos_limpios = df_procesado['Comentario'].astype(str).str.lower().str.strip()
                    
                    # Aplicar predicciones
                    df_procesado['Pred_Area'] = modelos['Area'].predict(textos_limpios)
                    df_procesado['Pred_Tipo'] = modelos['Tipo'].predict(textos_limpios)
                    df_procesado['Pred_Sentimiento'] = modelos['Sentimiento'].predict(textos_limpios)
                    df_procesado['Pred_NPS'] = modelos['NPS'].predict(textos_limpios)
                    
                    st.dataframe(df_procesado.head())
                    
                    # Botón de Descarga
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df_procesado.to_excel(writer, index=False)
                        
                    st.download_button(
                        label="Descargar Excel Tipificado",
                        data=buffer.getvalue(),
                        file_name="Encuestas_Tipificadas_IA.xlsx",
                        mime="application/vnd.ms-excel"
                    )
    elif archivo_nuevos and not modelos:
        st.warning("⚠️ Primero debes cargar el histórico y dar clic en 'Entrenar Modelos'.")

