"""Engaña al filtro de spam — reto del Hacking Game (IP San Sebastián).

Ejecutar localmente:  streamlit run app.py
"""
import time

import pandas as pd
import streamlit as st

import dashboard
import model
import storage

st.set_page_config(page_title="Engaña al filtro de spam", page_icon="🕵️", layout="centered")

MAX_CHARS = 280
COOLDOWN_S = 5


@st.cache_resource(show_spinner="Entrenando el filtro...")
def get_model():
    df = model.load_data()
    return model.train(df)


pipe, metrics = get_model()

st.title("🕵️ Engaña al filtro de spam")
st.caption(
    "Hacking Game · IP San Sebastián — Filtro con TF-IDF + regresión logística, "
    "entrenado con 1.000 correos simulados en español."
)

tab_reto, tab_dash, tab_detector, tab_info = st.tabs(
    ["🎯 Reto", "📊 Dashboard", "🔍 Detector", "ℹ️ Cómo funciona"]
)

# ------------------------------------------------------------------ Reto
with tab_reto:
    st.markdown(
        "**Objetivo:** escribe un mensaje que **siga siendo spam** y que el filtro clasifique como normal.\n\n"
        "**Reglas:** incluir un **link** y al menos una **palabra gancho** (premio, gratis, ganaste, sorteo, bono...). "
        "Gana quien logre la **menor probabilidad de spam**."
    )
    with st.form("reto", clear_on_submit=False):
        alias = st.text_input("Tu alias", max_chars=30, placeholder="ej: hacker_ipss")
        mensaje = st.text_area("Tu mensaje", max_chars=MAX_CHARS, height=110,
                               placeholder="ej: Hola profe, te dejo un pr3m1o en bit.ly/...")
        enviado = st.form_submit_button("Atacar al filtro 🚀", type="primary")

    if enviado:
        alias, mensaje = alias.strip(), mensaje.strip()
        last = st.session_state.get("last_submit", 0.0)
        if not alias or not mensaje:
            st.warning("Completa tu alias y tu mensaje.")
        elif time.time() - last < COOLDOWN_S:
            st.warning(f"Espera {COOLDOWN_S} segundos entre intentos.")
        elif problemas := model.check_rules(mensaje):
            st.error("Tu mensaje no cumple las reglas:\n\n- " + "\n- ".join(problemas))
        else:
            st.session_state["last_submit"] = time.time()
            p = model.predict_proba(pipe, mensaje)
            try:
                storage.save_attempt(alias, mensaje, p)
            except Exception as e:  # el reto sigue funcionando aunque falle el guardado
                st.warning(f"No pude guardar tu intento en el ranking ({type(e).__name__}).")
            st.metric("Probabilidad de spam", f"{p:.1%}")
            if p < 0.5:
                st.success("¡Lo lograste! El filtro lo dejó pasar como mensaje normal. 🎉")
                st.balloons()
            else:
                st.error("El filtro te detectó. Mira qué palabras te delataron y vuelve a intentarlo.")
            st.dataframe(model.explain(pipe, mensaje).round(3), hide_index=True, width="stretch")
            st.caption("Aporte positivo = empuja a spam · negativo = empuja a normal.")

    st.subheader("🏆 Ranking")
    try:
        intentos = storage.load_attempts()
    except Exception as e:
        intentos = pd.DataFrame(columns=storage.COLUMNS)
        st.warning(f"No pude leer el ranking ({type(e).__name__}).")
    if intentos.empty:
        st.info("Aún no hay intentos. ¡Sé el primero!")
    else:
        best = (intentos.sort_values(["prob_spam", "fecha"])
                .drop_duplicates("alias")
                .head(10)
                .reset_index(drop=True))
        best.index = best.index + 1
        best["prob_spam"] = (best["prob_spam"] * 100).round(1).astype(str) + "%"
        st.dataframe(best[["alias", "prob_spam", "mensaje"]], width="stretch")
        st.caption(f"{len(intentos)} intentos · {intentos['alias'].nunique()} participantes · "
                   f"guardado en {storage.backend_name()}")

# ------------------------------------------------------------- Dashboard
with tab_dash:
    dashboard.render(intentos, pipe)

# -------------------------------------------------------------- Detector
with tab_detector:
    st.markdown("Pega cualquier mensaje y mira cómo lo clasifica el filtro (no cuenta para el ranking).")
    texto = st.text_area("Mensaje", height=110, key="detector",
                         value="Felicidades, ganaste un premio de $500.000. Reclámalo en bit.ly/premio")
    if texto.strip():
        p = model.predict_proba(pipe, texto)
        st.metric("Probabilidad de spam", f"{p:.1%}", delta="SPAM" if p >= 0.5 else "normal",
                  delta_color="inverse" if p >= 0.5 else "normal")
        st.dataframe(model.explain(pipe, texto).round(3), hide_index=True, width="stretch")

# ------------------------------------------------------------------ Info
with tab_info:
    st.markdown(
        """
**1. TF-IDF** convierte cada mensaje en números: cuánto pesa cada palabra en ese mensaje (TF) y qué tan rara es en
el conjunto (IDF).

**2. La regresión logística** suma esos números multiplicados por un coeficiente por palabra y entrega una
probabilidad de spam. Sobre 50%, el mensaje es spam.

**3. El punto ciego:** el modelo solo conoce las palabras que vio al entrenar. Lo que no conoce, lo ignora. Y un link
como `bit.ly/premio` no lo lee como link: lo parte en `bit`, `ly` y `premio`.

**4. Análisis de anomalías:** los mensajes que logran pasar son la mejor información para mejorar el filtro.
"""
    )
    st.subheader("Palabras con más peso")
    tw = model.top_words(pipe, 12)
    c1, c2 = st.columns(2)
    c1.markdown("**Empujan a spam**")
    c1.dataframe(tw[tw.tipo == "spam"][["palabra", "coef"]].round(2), hide_index=True, width="stretch")
    c2.markdown("**Empujan a normal**")
    c2.dataframe(tw[tw.tipo == "normal"][["palabra", "coef"]].round(2), hide_index=True, width="stretch")
    st.warning(
        f"Exactitud en test: {metrics['accuracy']:.0%} con {metrics['n_total']} correos simulados "
        f"({metrics['n_spam']} spam / {metrics['n_normal']} normales). Un 100% con datos de plantillas no dice "
        "nada del mundo real: por eso lo atacamos."
    )
    st.caption("Proyecto de Fernando Pinto Montecinos · código en GitHub: fepintom/spam-hacking-game")
