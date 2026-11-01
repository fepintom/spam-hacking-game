# 🕵️ Engaña al filtro de spam

Reto interactivo de **Adversarial ML** presentado en el Hacking Game del IP San Sebastián, en la charla
*"Análisis de anomalías: cómo hackear un modelo para mejorarlo"*.

Un filtro de spam (**TF-IDF + regresión logística**) entrenado con 1.000 correos simulados en español. Tu misión:
escribir un mensaje que **siga siendo spam** y que el filtro deje pasar como normal. Los mensajes que logran
escaparse son las anomalías que nos dicen dónde está el punto ciego del modelo.

👉 **Pruébalo aquí:** _[enlace de Streamlit]_

## Reglas del reto
1. El mensaje debe incluir un **link** (`bit.ly/...`, `www.algo.cl`, etc.).
2. Debe tener al menos una **palabra gancho**: premio, gratis, ganaste, sorteo, bono, oferta... (vale escribirla
   como `pr3m1o` o `p.r.e.m.i.o`).
3. Gana quien logre la **menor probabilidad de spam**.

## Estructura
```
app.py                     App de Streamlit (reto + detector + explicación)
model.py                   Entrenamiento, predicción, explicación por palabra y reglas del reto
storage.py                 Ranking en Google Sheets (o CSV local si no hay credenciales)
data/correos_es.csv        1.000 correos simulados (400 spam / 600 normal)
data/generar_correos.py    Script que genera el dataset
notebooks/charla_spam.ipynb  Notebook de la charla: TF-IDF paso a paso, coeficientes,
                             análisis de los ataques y reentrenamiento
```

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
Sin credenciales, el ranking se guarda en `data/intentos_local.csv`.

## Desplegar en Streamlit Community Cloud
1. Entra a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.
2. **Create app** → elige este repositorio, rama `main` y archivo `app.py` → **Deploy**.

### Ranking persistente con Google Sheets
En Streamlit Cloud los archivos locales se borran cuando la app se reinicia, así que el ranking va a una planilla:

1. En [Google Cloud Console](https://console.cloud.google.com) crea un proyecto y habilita **Google Sheets API**.
2. **IAM y administración → Cuentas de servicio → Crear cuenta de servicio**. En la cuenta creada:
   **Claves → Agregar clave → JSON**. Se descarga un archivo con las credenciales.
3. Crea una planilla de Google Sheets vacía y **compártela como Editor** con el `client_email` de ese JSON.
4. En Streamlit Cloud: **App → Settings → Secrets**, usa la plantilla de
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example): la URL de la planilla y el contenido
   completo del JSON pegado tal cual entre `'''`.

## Advertencia honesta
Los datos son **simulados** con plantillas: el 100% de exactitud en test **no dice nada del mundo real**. El objetivo
es mostrar cómo un modelo que solo aprende palabras puede ser engañado (leetspeak, letras separadas, palabras que
diluyen la señal, links partidos en tokens) y cómo usar esos errores para mejorarlo.

---
Fernando Pinto Montecinos · Data Scientist · [GitHub](https://github.com/fepintom)
