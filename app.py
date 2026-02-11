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
st.set_page_config(page_title="Tipificador IA Hotelero", layout="wide")

# --- ENCABEZADO ---
col_logo, col_titulo = st.columns([1, 4])
with col_titulo:
    st.title("Sistema de Inteligencia Artificial")
    st.subheader("Tipificación Automática (Separa por guiones '-')")

# --- FUNCIONES DE LIMPIEZA Y PREPARACIÓN ---
def limpiar_texto_simple(texto):
    """Limpia espacios y minúsculas básicas"""
    if pd.isna(texto): return ""
    return str(texto).lower().strip()

def procesar_separacion_guiones(df, col_comentario):
    """
    1. Separa los comentarios por el guion '-'
    2. Crea nuevas filas (Explode)
    3. Filtra fragmentos vacíos o sin palabras reales
    """
    # Crear copia para no afectar el original inmediatamente
    df_exp = df.copy()
    
    # Asegurar que sea string
    df_exp[col_comentario] = df_exp[col_comentario].astype(str)
    
    # 1. SEPARAR: Convertir "Hola - Mundo" en ["Hola ", " Mundo"]
    df_exp[col_comentario] = df_exp[col_comentario].str.split('-')
    
    # 2. EXPLOTAR: Crear una fila por cada elemento de la lista
    df_exp = df_exp.explode(col_comentario)
    
    # 3. LIMPIEZA PROFUNDA DE FRAGMENTOS
    # Quitar espacios al inicio y final de cada fragmento
    df_exp[col_comentario] = df_exp[col_comentario].str.strip()
    
    # Filtrar: Solo nos quedamos con fragmentos que tengan al menos 2 letras
    # Esto elimina automáticamente los "--" (que se vuelven vacíos) o los "- -"
    df_exp = df_exp[df_exp[col_comentario].str.len() > 1]
    
    # Resetear el índice para mantener orden limpio
    df_exp.reset_index(drop=True, inplace=True)
    
    return df_exp

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
                st.toast(f"Usando columna '{posibles[0]}' como Comentario")
                df.rename(columns={posibles[0]: 'Comentario'}, inplace=True)
            else:
                st.error("❌ No encontré la columna 'Comentario'.")
                return None
        return df
    except Exception as e:
        st.error(f"Error leyendo archivo: {e}")
        return None

# --- ENTRENAMIENTO (LINEAR SVC - CEREBRO RÁPIDO) ---
@st.cache_resource
def entrenar_modelos(df_train):
    with st.spinner('Entrenando cerebro digital...'):
        # Limpieza básica para entrenamiento
        df = df_train.copy()
        df['clean_text'] = df['Comentario'].apply(limpiar_texto_simple)
        
        # Filtro de basura para entrenamiento
        stop_phrases = ['no', 'no.', 'ninguno', 'ninguna', 'sin comentarios', 'ok', 'na', 'no aplica']
        df = df[~df['clean_text'].isin(stop_phrases)]
        df = df[df['clean_text'].str.len() > 3]

        targets = {
            'Area': df['Area'], 
            'Tipo': df['Tipo'], 
            'Sentimiento': df['Clasificación'], 
            'NPS': df['Clasificación NPS']
        }
        
        modelos = {}
        metricas = {}

        for nombre, y in targets.items():
            # Pipeline Robusto
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

# BARRA LATERAL (Entrenamiento)
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
        st.progress(st.session_state['metricas']['NPS'], text=f"NPS: {st.session_state['metricas']['NPS']:.0%}")

# PANTALLA PRINCIPAL (Predicción)
st.write("Sube el archivo de encuestas. Si un comentario tiene guiones (`-`), se separará en varias filas.")

archivo_predecir = st.file_uploader("2. Sube Nuevas Encuestas", type=["csv", "xlsx"], key="pred")

if archivo_predecir and 'modelos' in st.session_state:
    df_new = cargar_archivo_inteligente(archivo_predecir)
    
    if df_new is not None:
        if st.button("Procesar y Tipificar 🚀"):
            # 1. SEPARACIÓN POR GUIONES (Aquí ocurre la magia)
            st.info(f"Filas originales: {len(df_new)}")
            df_expandido = procesar_separacion_guiones(df_new, 'Comentario')
            st.info(f"Filas después de separar por guiones (-): {len(df_expandido)}")
            
            # 2. PREPARAR TEXTO
            textos_limpios = df_expandido['Comentario'].apply(limpiar_texto_simple)
            
            # 3. PREDECIR
            modelos = st.session_state['modelos']
            df_expandido['Pred_Area'] = modelos['Area'].predict(textos_limpios)
            df_expandido['Pred_Tipo'] = modelos['Tipo'].predict(textos_limpios)
            df_expandido['Pred_Sentimiento'] = modelos['Sentimiento'].predict(textos_limpios)
            df_expandido['Pred_NPS'] = modelos['NPS'].predict(textos_limpios)
            
            # 4. MOSTRAR RESULTADOS
            st.dataframe(df_expandido.head(10))
            
            # 5. DESCARGAR
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_expandido.to_excel(writer, index=False)
            
            st.download_button(
                label="Descargar Excel Final", 
                data=buffer.getvalue(), 
                file_name="Tipificacion_Separada.xlsx",
                mime="application/vnd.ms-excel"
            )

elif archivo_predecir and 'modelos' not in st.session_state:
    st.warning("⚠️ Recuerda entrenar el modelo primero en el menú de la izquierda.")
















