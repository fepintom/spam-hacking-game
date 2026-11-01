"""Modelo del filtro de spam: TF-IDF + regresión logística.

Se entrena con data/correos_es.csv (correos simulados en español).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).parent / "data" / "correos_es.csv"

TEXT_COLS = ["texto", "mensaje", "correo", "text", "message", "email"]
LABEL_COLS = ["etiqueta", "label", "clase", "categoria", "tipo", "spam", "target"]
SPAM_VALUES = {"spam", "1", "si", "sí", "true", "yes"}

STOPWORDS_ES = [
    "a", "al", "algo", "ante", "antes", "aqui", "aquí", "asi", "así", "aun", "con",
    "como", "cual", "cuando", "de", "del", "desde", "donde", "dos", "e", "el", "ella",
    "ellos", "en", "entre", "era", "es", "esa", "ese", "eso", "esta", "está", "estas",
    "este", "esto", "estos", "fue", "ha", "hay", "la", "las", "le", "les", "lo", "los",
    "mas", "más", "me", "mi", "mis", "muy", "ni", "no", "nos", "o", "os", "para",
    "pero", "por", "porque", "que", "qué", "se", "sea", "si", "sin", "sobre", "son",
    "su", "sus", "te", "ti", "tu", "tus", "un", "una", "uno", "unos", "unas", "y", "ya",
    "yo", "es", "ser", "fue", "han", "hasta",
]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise ValueError(
        f"No encontré ninguna de estas columnas en el CSV: {candidates}. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = _find_col(df, TEXT_COLS)
    label_col = _find_col(df, LABEL_COLS)
    out = pd.DataFrame(
        {
            "texto": df[text_col].astype(str),
            "spam": df[label_col].astype(str).str.lower().str.strip().isin(SPAM_VALUES).astype(int),
        }
    )
    return out.dropna().reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words=STOPWORDS_ES)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train(df: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["spam"], test_size=0.25, random_state=seed, stratify=df["spam"]
    )
    pipe = build_pipeline().fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "n_total": len(df),
        "n_spam": int(df["spam"].sum()),
        "n_normal": int((df["spam"] == 0).sum()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "confusion": confusion_matrix(y_test, pred).tolist(),
    }
    # Modelo final con todos los datos, para que el reto sea lo más difícil posible.
    final = build_pipeline().fit(df["texto"], df["spam"])
    return final, metrics


def predict_proba(pipe: Pipeline, text: str) -> float:
    return float(pipe.predict_proba([text])[0, 1])


def explain(pipe: Pipeline, text: str, top: int = 10) -> pd.DataFrame:
    """Aporte de cada palabra del mensaje: valor TF-IDF × coeficiente."""
    tfidf: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    vec = tfidf.transform([text])
    vocab = np.array(tfidf.get_feature_names_out())
    idx = vec.nonzero()[1]
    if len(idx) == 0:
        return pd.DataFrame(columns=["palabra", "aporte"])
    contrib = vec.toarray()[0, idx] * clf.coef_[0, idx]
    df = pd.DataFrame({"palabra": vocab[idx], "aporte": contrib})
    return df.reindex(df["aporte"].abs().sort_values(ascending=False).index).head(top)


def vocabulary(pipe: Pipeline) -> dict:
    return pipe.named_steps["tfidf"].vocabulary_


def top_words(pipe: Pipeline, n: int = 15) -> pd.DataFrame:
    tfidf: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    vocab = np.array(tfidf.get_feature_names_out())
    coef = clf.coef_[0]
    order = np.argsort(coef)
    spam = pd.DataFrame({"palabra": vocab[order[-n:][::-1]], "coef": coef[order[-n:][::-1]], "tipo": "spam"})
    normal = pd.DataFrame({"palabra": vocab[order[:n]], "coef": coef[order[:n]], "tipo": "normal"})
    return pd.concat([spam, normal], ignore_index=True)


# --- Reglas del reto -------------------------------------------------------

LINK_RE = re.compile(r"(https?://|www\.|bit\.ly|\b[\w-]+\.(cl|com|net|org|link|xyz|info)\b)", re.I)
BAIT_WORDS = [
    "premio", "gratis", "ganaste", "ganador", "oferta", "dinero", "regalo", "sorteo",
    "urgente", "bono", "descuento", "promocion", "promoción", "clic", "click", "cuenta",
    "transferencia", "credito", "crédito", "prestamo", "préstamo", "reclama", "cupon", "cupón",
]


LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def _normalize(text: str) -> str:
    """Normalización solo para revisar las reglas: deshace leetspeak y separadores.

    Así 'pr3m1o' o 'p.r.e.m.i.o' cuentan como palabra gancho (el modelo NO
    hace esta normalización; ese es justamente su punto ciego).
    """
    t = text.lower().translate(LEET)
    t = t.translate(str.maketrans("áéíóú", "aeiou"))
    return re.sub(r"[^a-zñ]", "", t)


def check_rules(text: str) -> list[str]:
    """Devuelve la lista de reglas que el mensaje NO cumple (vacía = válido)."""
    problems = []
    if not LINK_RE.search(text):
        problems.append("Debe incluir un link (por ejemplo, bit.ly/premio o www.algo.cl).")
    normalized = _normalize(text)
    bait = {_normalize(w) for w in BAIT_WORDS}
    if not any(w in normalized for w in bait):
        problems.append(
            "Debe seguir siendo spam: incluye al menos una palabra gancho "
            "(premio, gratis, ganaste, oferta, sorteo, bono, etc.)."
        )
    if len(text.split()) < 4:
        problems.append("Escribe un mensaje de al menos 4 palabras.")
    return problems
