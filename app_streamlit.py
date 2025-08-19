# app_streamlit.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import numpy as np

st.set_page_config(page_title="Clasificador de Comentarios", layout="wide")

st.title("Clasificador de Comentarios — Tags y Sentiment")

# --- Cargar datos base ---
@st.cache_data
def load_base():
    try:
        df = pd.read_csv("base_dataset.csv")
        return df
    except Exception:
        return None

base_df = load_base()
uploaded = st.sidebar.file_uploader("Carga un CSV con columnas: Corpo Descricao, sentiment, Tags", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        df = pd.read_csv(uploaded, encoding="latin1")
else:
    df = base_df

if df is None:
    st.warning("Sube un CSV para entrenar el modelo.")
    st.stop()

expected_cols = ["Corpo Descricao", "sentiment", "Tags"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.error(f"Faltan columnas requeridas: {missing}")
    st.stop()

st.success(f"Dataset cargado con {len(df)} filas")

with st.expander("Vista previa del dataset"):
    st.dataframe(df.head(10))

# --- Preprocesamiento ---
def clean_text(s):
    return str(s) if pd.notna(s) else ""

def prepare_tags(tags_series):
    tag_lists = []
    for t in tags_series.fillna("").astype(str).tolist():
        parts = [p.strip() for p in t.split(",") if p.strip()]
        tag_lists.append(parts)
    return tag_lists

X_text = df["Corpo Descricao"].map(clean_text).values
y_sent = df["sentiment"].astype(str).str.upper().values
y_tags_raw = prepare_tags(df["Tags"])

# --- Modelo Sentiment ---
sent_clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000, stop_words="spanish")),
    ("clf", LogisticRegression(max_iter=200))
])

X_train, X_test, y_train, y_test = train_test_split(X_text, y_sent, test_size=0.2, random_state=42, stratify=y_sent)
sent_clf.fit(X_train, y_train)
y_pred = sent_clf.predict(X_test)

with st.expander("Reporte de Sentiment"):
    st.text(classification_report(y_test, y_pred, zero_division=0))

# --- Modelo Tags ---
mlb = MultiLabelBinarizer()
Y_tags = mlb.fit_transform(y_tags_raw)

tags_clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000, stop_words="spanish")),
    ("clf", OneVsRestClassifier(LogisticRegression(max_iter=200)))
])

X_train_t, X_test_t, Y_train_t, Y_test_t = train_test_split(X_text, Y_tags, test_size=0.2, random_state=42)
tags_clf.fit(X_train_t, Y_train_t)
Y_pred = tags_clf.predict(X_test_t)

micro_p = precision_score(Y_test_t, Y_pred, average="micro", zero_division=0)
micro_r = recall_score(Y_test_t, Y_pred, average="micro", zero_division=0)
micro_f1 = f1_score(Y_test_t, Y_pred, average="micro", zero_division=0)

st.sidebar.metric("Tags aprendidos", len(mlb.classes_))
st.sidebar.metric("F1 micro (tags)", f"{micro_f1:.2f}")

# --- Uso manual ---
st.header("Etiquetar un comentario manualmente")
text = st.text_area("Escribe un comentario")
if st.button("Predecir"):
    if text.strip():
        proba = sent_clf.predict_proba([text])[0]
        sent_labels = sent_clf.classes_
        pred_sent = sent_labels[int(np.argmax(proba))]
        st.subheader("Sentiment")
        st.write(pred_sent)
        st.progress(float(np.max(proba)))

        tag_scores = tags_clf.decision_function([text])[0]
        top_idx = np.argsort(tag_scores)[::-1][:5]
        st.subheader("Tags sugeridos")
        st.write([mlb.classes_[i] for i in top_idx])

# --- Etiquetado masivo ---
st.header("Etiquetado masivo")
upload_mass = st.file_uploader("Carga un CSV para etiquetado masivo", type=["csv"], key="mass")
if upload_mass:
    try:
        df_new = pd.read_csv(upload_mass)
    except Exception:
        df_new = pd.read_csv(upload_mass, encoding="latin1")

    if "Corpo Descricao" not in df_new.columns:
        st.error("El CSV debe contener la columna 'Corpo Descricao'")
    else:
        texts = df_new["Corpo Descricao"].map(clean_text).tolist()
        sent_pred = sent_clf.predict(texts)
        sent_proba = sent_clf.predict_proba(texts)
        tag_scores = tags_clf.decision_function(texts)

        best_sent = []
        best_conf = []
        best_tags = []
        for i in range(len(texts)):
            idx = int(np.argmax(sent_proba[i]))
            best_sent.append(sent_clf.classes_[idx])
            best_conf.append(float(np.max(sent_proba[i])))
            top_idx = np.argsort(tag_scores[i])[::-1][:5]
            best_tags.append(", ".join([mlb.classes_[j] for j in top_idx]))

        df_new["sentiment_pred"] = best_sent
        df_new["sentiment_confidence"] = best_conf
        df_new["tags_sugeridos"] = best_tags

        st.dataframe(df_new.head(20))
        st.download_button("Descargar resultados", data=df_new.to_csv(index=False).encode("utf-8-sig"), file_name="resultados.csv")
