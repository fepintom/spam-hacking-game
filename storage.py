"""Guardado de intentos del reto.

- Si existen credenciales en st.secrets (Streamlit Cloud), usa Google Sheets.
- Si no, guarda en un CSV local (útil para probar en tu computador).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

COLUMNS = ["fecha", "alias", "mensaje", "prob_spam", "evadio"]
LOCAL_PATH = Path(__file__).parent / "data" / "intentos_local.csv"


def _sheets_available() -> bool:
    try:
        return "gcp_service_account" in st.secrets and "sheet_url" in st.secrets
    except Exception:  # No hay secrets.toml
        return False


@st.cache_resource(show_spinner=False)
def _worksheet():
    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    sheet = gspread.authorize(creds).open_by_url(st.secrets["sheet_url"])
    ws = sheet.sheet1
    if ws.row_values(1) != COLUMNS:
        ws.update(range_name="A1:E1", values=[COLUMNS])
    return ws


def backend_name() -> str:
    return "Google Sheets" if _sheets_available() else "archivo local"


def save_attempt(alias: str, mensaje: str, prob_spam: float) -> None:
    row = [
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        alias,
        mensaje,
        round(prob_spam, 4),
        int(prob_spam < 0.5),
    ]
    if _sheets_available():
        _worksheet().append_row(row, value_input_option="RAW")
    else:
        LOCAL_PATH.parent.mkdir(exist_ok=True)
        new = pd.DataFrame([row], columns=COLUMNS)
        new.to_csv(LOCAL_PATH, mode="a", header=not LOCAL_PATH.exists(), index=False)
    load_attempts.clear()


@st.cache_data(ttl=15, show_spinner=False)
def load_attempts() -> pd.DataFrame:
    if _sheets_available():
        records = _worksheet().get_all_records()
        df = pd.DataFrame(records, columns=COLUMNS)
    elif LOCAL_PATH.exists():
        df = pd.read_csv(LOCAL_PATH)
    else:
        df = pd.DataFrame(columns=COLUMNS)
    if not df.empty:
        df["prob_spam"] = pd.to_numeric(df["prob_spam"], errors="coerce")
        df["evadio"] = pd.to_numeric(df["evadio"], errors="coerce").fillna(0).astype(int)
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df
