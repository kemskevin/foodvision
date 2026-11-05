import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="FoodVision", page_icon="🍽️", layout="centered")
st.title("🍽️ FoodVision")
st.caption("Classification de plats (Food-101) — modèle EfficientNet-B0 (ONNX) servi par FastAPI")

with st.sidebar:
    st.header("Connexion")
    api_url = st.text_input("URL de l'API", DEFAULT_API_URL)
    token = st.text_input("Clé API (X-API-Key)", type="password", value=os.getenv("DEFAULT_API_KEY", ""))
    if st.button("Voir mon usage", disabled=not token):
        try:
            r = requests.get(f"{api_url}/me/usage", headers={"X-API-Key": token}, timeout=10)
            if r.ok:
                st.json(r.json())
            else:
                st.error(f"{r.status_code} : {r.json().get('detail', r.text)}")
        except requests.RequestException as exc:
            st.error(f"API injoignable : {exc}")
    try:
        health = requests.get(f"{api_url}/health", timeout=5).json()
        st.success(f"API OK — modèle {health['model_version']}")
    except Exception:
        st.warning("API injoignable")

uploaded = st.file_uploader("Choisis une photo de plat", type=["jpg", "jpeg", "png", "webp"])
top_k = st.slider("Nombre de prédictions", 1, 10, 5)

if uploaded:
    st.image(uploaded, width=450)
    if st.button("Prédire", type="primary", disabled=not token):
        with st.spinner("Analyse en cours…"):
            try:
                r = requests.post(
                    f"{api_url}/predict",
                    params={"top_k": top_k},
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    headers={"X-API-Key": token},
                    timeout=30,
                )
            except requests.RequestException as exc:
                st.error(f"API injoignable : {exc}")
                st.stop()
        if r.status_code == 401:
            st.error("Clé API invalide ou manquante.")
        elif not r.ok:
            st.error(f"Erreur {r.status_code} : {r.json().get('detail', r.text)}")
        else:
            body = r.json()
            best = body["predictions"][0]
            st.subheader(f"{best['label'].replace('_', ' ').title()} — {best['score']:.1%}")
            df = pd.DataFrame(body["predictions"])
            df["label"] = df["label"].str.replace("_", " ")
            chart = (
                alt.Chart(df)
                .mark_bar(color="#E4572E")
                .encode(
                    x=alt.X("score:Q", title="Probabilité", axis=alt.Axis(format="%")),
                    y=alt.Y("label:N", sort="-x", title=None),
                    tooltip=["label", alt.Tooltip("score:Q", format=".1%")],
                )
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption(f"Modèle {body['model_version']} · déploiement {body['deployment']} · "
                       f"{body['latency_ms']} ms")
    elif not token:
        st.info("Renseigne ta clé API dans la barre latérale.")
