import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io
st.set_page_config(page_title="Tipificador Zuana", layout="wide")
col_logo, col_titulo = st.columns([1, 4])

with col_logo:
    st.image("Logo.png", width=150) 

with col_titulo:
    st.title("🏨 Tipificador Encuestas Experiencia Hotel Zuana 🏨")
    st.subheader("Análisis de Comentarios")
st.markdown("""
Este modelo interpretativo tiene como objetivo **evaluar** comentarios realizados por Huespedes del hotel Zuana.
Se aclara que si la IA detecta que un cliente habla de varias cosas a la vez, duplicará la fila automáticamente.
""")

def limpiar_texto(df, col_comentario):
    df['clean_text'] = df[col_comentario].astype(str).str.lower().str.strip()
    stop_phrases = ['no', 'no.', 'ninguno', 'ninguna', 'sin comentarios', 'ok', 'na', 'no aplica', 'todo bien', 'gracias']
    df_clean = df[~df['clean_text'].isin(stop_phrases)].copy()
    df_clean = df_clean[df_clean['clean_text'].str.len() > 3]
    return df_clean
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

@st.cache_resource
def entrenar_modelos(df_train):
    with st.spinner('Entrenando cerebro con capacidad de probabilidades...'):
        required = ['Area', 'Tipo', 'Clasificación', 'Clasificación NPS']
        if not all(col in df_train.columns for col in required):
            st.error(f"Faltan columnas. Requeridas: {required}")
            return None, None

        df = limpiar_texto(df_train, 'Comentario')
        targets = {'Area': df['Area'], 'Tipo': df['Tipo'], 'Sentimiento': df['Clasificación'], 'NPS': df['Clasificación NPS']}
        
        modelos = {}
        metricas = {}

        for nombre, y in targets.items():
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))), 
                ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
            ])
            
            X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            modelos[nombre] = pipeline
            metricas[nombre] = pipeline.score(X_test, y_test)
            
        return modelos, metricas
def predecir_multilabel(model, textos, umbral=0.3):
    """
    Devuelve una lista de tuplas (indice_original, etiqueta_predicha)
    Si la probabilidad supera el umbral, se agrega.
    """
    probs = model.predict_proba(textos)
    clases = model.classes_
    resultados_expandidos = []
    
    for i, prob_row in enumerate(probs):
        indices_validos = np.where(prob_row >= umbral)[0]
        
        if len(indices_validos) == 0:
            indices_validos = [np.argmax(prob_row)]
            
        for idx in indices_validos:
            resultados_expandidos.append({
                'indice_orig': i,
                'Prediccion': clases[idx],
                'Confianza': prob_row[idx]
            })
            
    return pd.DataFrame(resultados_expandidos)
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Cargar Historico 📉")
    archivo_entrenar = st.file_uploader("Sube Histórico", type=["csv", "xlsx"], key="train")
    if archivo_entrenar:
        df_train = cargar_archivo_inteligente(archivo_entrenar)
        if df_train is not None and st.button("Entrenar"):
            modelos, metricas = entrenar_modelos(df_train)
            st.session_state['modelos'] = modelos
            st.session_state['metricas'] = metricas

if 'modelos' in st.session_state:
    st.sidebar.success(f"Modelos Activos. Acc: {st.session_state['metricas']['Area']:.1%}")

with col2:
    st.header("2. Cargar Base a tipificar 📉")
    archivo_predecir = st.file_uploader("Sube Nuevas Encuestas", type=["csv", "xlsx"], key="pred")

    umbral = st.slider("Sensibilidad de Duplicación (Umbral)", 0.1, 0.9, 0.30, 
                       help="Si bajas esto, duplicará más filas (detectará más temas tenues). Si lo subes, solo duplicará lo muy obvio.")

    if archivo_predecir and 'modelos' in st.session_state:
        df_new = cargar_archivo_inteligente(archivo_predecir)
        
        if df_new is not None:
            if st.button("Tipificar y Expandir"):
                textos = df_new['Comentario'].astype(str).str.lower().str.strip()
                modelo_area = st.session_state['modelos']['Area']
                df_areas_expandidas = predecir_multilabel(modelo_area, textos, umbral=umbral)
                df_final = df_areas_expandidas.merge(df_new, left_on='indice_orig', right_index=True)
                df_final = df_final.rename(columns={'Prediccion': 'Pred_Area'})
                textos_expandidos = df_final['Comentario'].astype(str).str.lower().str.strip()
                
                df_final['Pred_Tipo'] = st.session_state['modelos']['Tipo'].predict(textos_expandidos)
                df_final['Pred_Sentimiento'] = st.session_state['modelos']['Sentimiento'].predict(textos_expandidos)
                df_final['Pred_NPS'] = st.session_state['modelos']['NPS'].predict(textos_expandidos)
                
                cols_orden = ['Comentario', 'Pred_Area', 'Pred_Tipo', 'Pred_Sentimiento', 'Pred_NPS', 'Confianza']
                otras_cols = [c for c in df_final.columns if c not in cols_orden and c != 'indice_orig']
                df_final = df_final[cols_orden + otras_cols]
                
                st.write(f"De {len(df_new)} comentarios originales, se generaron {len(df_final)} filas tipificadas.")
                st.dataframe(df_final.head())
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_final.to_excel(writer, index=False)
                st.download_button("Descargar Excel Multi-Etiqueta", buffer.getvalue(), "Tipificacion_Expandida.xlsx")













