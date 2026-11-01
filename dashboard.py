"""Pestaña de dashboard de la competencia (Streamlit + Altair)."""
from __future__ import annotations

import re
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

import model

GREEN, RED, GREY = "#22c55e", "#ef4444", "#94a3b8"
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def _tecnicas(msg: str, vocab: dict) -> list[str]:
    """Detección aproximada de técnicas de evasión."""
    out = []
    if re.search(r"[a-záéíóúñ][013457@$][a-záéíóúñ0-9]|[013457@$][a-záéíóúñ]{2,}", msg.lower()):
        out.append("Leetspeak")
    if re.search(r"\b(\w[ .\-_]){3,}\w\b", msg):
        out.append("Letras separadas")
    tokens = [t for t in TOKEN_RE.findall(msg.lower()) if t in vocab]
    if len(tokens) >= 8:
        out.append("Dilución")
    return out or ["Otra"]


def render(intentos: pd.DataFrame, pipe) -> None:
    if intentos.empty:
        st.info("Cuando lleguen los primeros intentos, aquí verás las estadísticas de la competencia.")
        return

    df = intentos.dropna(subset=["prob_spam"]).copy()
    df["resultado"] = df["evadio"].map({1: "Evadió", 0: "Detectado"})
    vocab = model.vocabulary(pipe)

    # --- Tarjetas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intentos", f"{len(df):,}".replace(",", "."))
    c2.metric("Participantes", df["alias"].nunique())
    c3.metric("Tasa de evasión", f"{df['evadio'].mean():.0%}")
    c4.metric("Mejor puntaje", f"{df['prob_spam'].min():.1%}")

    # --- Distribución
    st.markdown("#### ¿Qué tan cerca quedan los ataques?")
    hist = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("prob_spam:Q", bin=alt.Bin(step=0.05), title="Probabilidad de spam",
                    axis=alt.Axis(format="%")),
            y=alt.Y("count():Q", title="Intentos"),
            color=alt.Color("resultado:N", scale=alt.Scale(domain=["Evadió", "Detectado"], range=[GREEN, RED]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("count():Q", title="Intentos")],
        )
    )
    umbral = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[6, 4], color=GREY).encode(x="x:Q")
    st.altair_chart((hist + umbral).properties(height=240), width="stretch")

    # --- En el tiempo
    st.markdown("#### Intentos en el tiempo")
    span_h = (df["fecha"].max() - df["fecha"].min()).total_seconds() / 3600
    freq, fmt = ("D", "%d %b") if span_h > 72 else ("h", "%d %b %H:%M")
    if span_h > 24 * 3:
        freq, fmt = "D", "%d %b"
    elif span_h > 12:
        freq, fmt = "3h", "%d %b %H:%M"
    ts = (df.assign(t=df["fecha"].dt.floor(freq))
          .pivot_table(index="t", columns="resultado", values="alias", aggfunc="count", fill_value=0)
          .reindex(columns=["Evadió", "Detectado"], fill_value=0))
    ts = ts.reindex(pd.date_range(ts.index.min(), ts.index.max(), freq=freq), fill_value=0)
    ts = ts.rename_axis("t").reset_index().melt("t", var_name="resultado", value_name="n")
    linea = (
        alt.Chart(ts)
        .mark_bar(size=22)
        .encode(
            x=alt.X("t:T", title=None, axis=alt.Axis(format=fmt)),
            y=alt.Y("n:Q", stack=True, title="Intentos"),
            color=alt.Color("resultado:N", scale=alt.Scale(domain=["Evadió", "Detectado"], range=[GREEN, RED]),
                            legend=None),
            tooltip=[alt.Tooltip("t:T", title="Hora", format=fmt), "resultado:N", alt.Tooltip("n:Q", title="Intentos")],
        )
        .properties(height=200)
    )
    st.altair_chart(linea, width="stretch")

    col_a, col_b = st.columns(2)

    # --- Palabras de los ataques exitosos
    with col_a:
        st.markdown("#### Palabras en ataques exitosos")
        cnt = Counter(
            t for m in df.loc[df["evadio"] == 1, "mensaje"]
            for t in set(TOKEN_RE.findall(str(m).lower())) if t not in model.STOPWORDS_ES
        )
        if cnt:
            todas = pd.DataFrame(cnt.most_common(), columns=["palabra", "n"])
            todas["el modelo"] = todas["palabra"].map(lambda w: "La conoce" if w in vocab else "No la conoce")
            # Las desconocidas son el punto ciego: siempre mostramos las más usadas de ese grupo.
            pw = pd.concat([todas[todas["el modelo"] == "La conoce"].head(7),
                            todas[todas["el modelo"] == "No la conoce"].head(5)])
            bars = (
                alt.Chart(pw)
                .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                .encode(
                    y=alt.Y("palabra:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
                    x=alt.X("n:Q", title="Ataques que la usan"),
                    color=alt.Color("el modelo:N", scale=alt.Scale(domain=["No la conoce", "La conoce"],
                                                                 range=["#f59e0b", GREY]),
                                    legend=alt.Legend(title=None, orient="bottom")),
                    tooltip=["palabra:N", "n:Q", "el modelo:N"],
                )
                .properties(height=26 * len(pw))
            )
            st.altair_chart(bars, width="stretch")
        else:
            st.caption("Aún no hay ataques exitosos.")

    # --- Técnicas
    with col_b:
        st.markdown("#### Técnicas en ataques exitosos")
        ok = df.loc[df["evadio"] == 1, "mensaje"]
        tec = Counter(t for m in ok for t in _tecnicas(str(m), vocab))
        if tec:
            td = pd.DataFrame(tec.items(), columns=["técnica", "n"])
            donut = (
                alt.Chart(td)
                .mark_arc(innerRadius=60)
                .encode(
                    theta="n:Q",
                    color=alt.Color("técnica:N", scale=alt.Scale(range=[GREEN, "#f59e0b", "#38bdf8", GREY]),
                                    legend=alt.Legend(title=None, orient="bottom", columns=2)),
                    tooltip=["técnica:N", alt.Tooltip("n:Q", title="Ataques")],
                )
                .properties(height=320)
            )
            st.altair_chart(donut, width="stretch")
            st.caption("Detección aproximada con reglas simples; un ataque puede usar varias técnicas. "
                       "Dilución = mezclar el spam con palabras de mensajes normales.")
        else:
            st.caption("Aún no hay ataques exitosos.")
