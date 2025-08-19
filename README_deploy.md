# Cómo publicar el mini‑sitio (Streamlit)

## A) Streamlit Community Cloud (rápido)
1. Sube a GitHub: `app_streamlit.py`, `base_dataset.csv`, `requirements.txt`, `.streamlit/config.toml` (opcional).
2. En https://share.streamlit.io/ conecta tu GitHub.
3. Selecciona el repo y `Main file` = `app_streamlit.py`.
4. Deploy y listo: tendrás una URL pública.

## B) Hugging Face Spaces
1. Crea un Space tipo **Streamlit**.
2. Sube los mismos archivos (y `requirements.txt`).
3. Se construye y publica la app automáticamente.

## C) Render/Railway (con Procfile o Docker)
- **Procfile** (incluido): `web: streamlit run app_streamlit.py --server.port $PORT --server.address 0.0.0.0`
- **Docker** (incluido): despliega el contenedor y expone el puerto.

## D) Docker local/VPS
```bash
docker build -t comentarios-app .
docker run -p 8501:8501 comentarios-app
# http://localhost:8501
```

### Tips
- Para CSVs grandes y edición persistente, conecta BD (SQLite/Postgres) y agrega auth básica.
- Las visualizaciones y wordcloud ya están consideradas en `requirements.txt`.
