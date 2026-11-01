"""Construye notebooks/charla_spam.ipynb (se ejecuta una vez; el notebook es el entregable)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Análisis de anomalías: cómo hackear un modelo para mejorarlo
**Hacking Game · IP San Sebastián** — Fernando Pinto Montecinos

Filtro de spam con **TF-IDF + regresión logística**. Lo entrenamos, lo atacamos entre todos y usamos sus errores
(las anomalías) para mejorarlo.

> **Ojo:** los datos son **simulados** (1.000 correos en español generados con plantillas). Los resultados son una
> demostración, no una medición del mundo real.
""")

code("""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

plt.rcParams["figure.figsize"] = (9, 4)
pd.set_option("display.max_colwidth", 120)
""")

md("## 1. Los datos")
code("""
df = pd.read_csv("../data/correos_es.csv")
df["spam"] = (df["etiqueta"] == "spam").astype(int)
print(df.shape)
print(df["etiqueta"].value_counts())
df.sample(8, random_state=1)
""")

md("""
## 2. ¿Cómo lee una máquina un correo? TF-IDF en 4 pasos
Ejemplo con tres mensajes. Usamos la fórmula de la charla: **TF** = veces que aparece la palabra / total de palabras
del mensaje, **IDF** = log(N / mensajes que contienen la palabra).
""")
code("""
mini = ["ganaste premio gratis", "premio gratis ya", "hola mamá llegas almorzar"]

# Paso 1: vocabulario
vocab = sorted({w for m in mini for w in m.split()})
print("Vocabulario:", vocab)

# Paso 2: TF
tf = pd.DataFrame([[m.split().count(w) / len(m.split()) for w in vocab] for m in mini],
                  columns=vocab, index=[f"m{i+1}" for i in range(len(mini))])

# Paso 3: IDF
N = len(mini)
idf = pd.Series({w: np.log(N / sum(w in m.split() for m in mini)) for w in vocab})

# Paso 4: TF × IDF
tfidf_manual = tf * idf
display(tf.round(2)); display(idf.round(2).to_frame("IDF").T); display(tfidf_manual.round(2))
""")
md("Esta matriz es lo que recibe el modelo. TF-IDF solo la construye; la regresión logística decide.")

md("## 3. Entrenamiento")
code("""
STOPWORDS_ES = ["a","al","algo","ante","antes","aqui","aquí","asi","así","con","como","cual","cuando","de","del",
    "desde","donde","e","el","ella","en","entre","era","es","esa","ese","eso","esta","está","este","esto","fue","ha",
    "hay","la","las","le","les","lo","los","mas","más","me","mi","mis","muy","ni","no","nos","o","para","pero","por",
    "porque","que","qué","se","si","sin","sobre","son","su","sus","te","tu","tus","un","una","uno","y","ya","yo"]

X_train, X_test, y_train, y_test = train_test_split(
    df["texto"], df["spam"], test_size=0.25, random_state=42, stratify=df["spam"])

vectorizer = TfidfVectorizer(lowercase=True, stop_words=STOPWORDS_ES)
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

modelo = LogisticRegression(max_iter=1000).fit(Xtr, y_train)
pred = modelo.predict(Xte)
print(f"Exactitud en test: {accuracy_score(y_test, pred):.1%}")
print(classification_report(y_test, pred, target_names=["normal", "spam"]))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["normal", "spam"], cmap="Greens");
""")
md("""
> **¿100% de exactitud? Desconfía.** Los correos salen de plantillas, así que los de entrenamiento y los de test se
> parecen demasiado. El número no dice nada sobre el mundo real. Por eso, en vez de mirar la métrica, vamos a
> **atacar** el modelo.
""")

md("## 4. ¿Qué aprendió? Los coeficientes")
code("""
vocabulario = np.array(vectorizer.get_feature_names_out())
coef = modelo.coef_[0]
orden = np.argsort(coef)
top = pd.concat([
    pd.DataFrame({"palabra": vocabulario[orden[-15:]], "coef": coef[orden[-15:]]}),
    pd.DataFrame({"palabra": vocabulario[orden[:15]], "coef": coef[orden[:15]]}),
]).sort_values("coef")

colores = ["#2e7d32" if c < 0 else "#c62828" for c in top["coef"]]
plt.figure(figsize=(8, 8))
plt.barh(top["palabra"], top["coef"], color=colores)
plt.axvline(0, color="grey", lw=0.8)
plt.title("Palabras que empujan a spam (rojo) y a normal (verde)")
plt.tight_layout(); plt.show()
""")

md("### Nubes de palabras según el peso en el modelo (no según la frecuencia)")
code("""
from wordcloud import WordCloud

pesos_spam = {w: c for w, c in zip(vocabulario, coef) if c > 0}
pesos_normal = {w: -c for w, c in zip(vocabulario, coef) if c < 0}

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for a, pesos, titulo, cmap in [(ax[0], pesos_spam, "Spam", "Reds"), (ax[1], pesos_normal, "Normal", "Greens")]:
    wc = WordCloud(width=700, height=400, background_color="white", colormap=cmap).generate_from_frequencies(pesos)
    a.imshow(wc); a.axis("off"); a.set_title(titulo, fontsize=16)
plt.tight_layout(); plt.show()
""")

md("## 5. Probemos mensajes")
code("""
def clasificar(texto, top_n=6):
    v = vectorizer.transform([texto])
    p = modelo.predict_proba(v)[0, 1]
    idx = v.nonzero()[1]
    aportes = sorted(zip(vocabulario[idx], v.toarray()[0, idx] * coef[idx]), key=lambda x: -abs(x[1]))[:top_n]
    ignoradas = [w for w in re.findall(r"(?u)\\b\\w\\w+\\b", texto.lower())
                 if w not in vectorizer.vocabulary_ and w not in STOPWORDS_ES]
    etiqueta = "SPAM  " if p >= 0.5 else "normal"
    print(f"{etiqueta}  P(spam) = {p:.1%}   →  {texto}")
    print("   aportes:", ", ".join(f"{w} ({a:+.2f})" for w, a in aportes) or "—")
    if ignoradas:
        print("   palabras que el modelo no conoce (las ignora):", ", ".join(ignoradas))
    return p

clasificar("Felicidades, ganaste un premio de $500.000. Reclámalo en bit.ly/premio")
clasificar("Hola Camila, ¿nos juntamos el lunes para revisar el informe?")
clasificar("Hola profe, gracias por la clase de estadística. Pr3m1o para ti en bit.ly/premio");
""")
md("""
Fíjate en el link: `bit.ly/premio` no se lee como un link. El tokenizador lo parte en `bit`, `ly` y `premio`.
El modelo no sabe qué es una URL; solo ve palabras sueltas. Ese es uno de sus puntos ciegos.
""")

md("""
## 6. El reto: análisis de anomalías
Descarga las respuestas del Google Form (**Respuestas → Descargar CSV**), guárdalas como `respuestas_form.csv`
junto a este notebook y ejecuta la celda. Si el archivo no existe, se usan mensajes de ejemplo.
""")
code("""
from pathlib import Path

ruta = Path("respuestas_form.csv")
if ruta.exists():
    form = pd.read_csv(ruta)
    col = [c for c in form.columns if "mensaje" in c.lower() or "spam" in c.lower()][0]
    ataques = form[col].dropna().astype(str).tolist()
else:
    ataques = [
        "Hola profe, gracias por la clase. Te dejo un pr3m1o en bit.ly/x9",
        "Camila, el informe de estadística quedó listo. Por cierto gan4ste un b0no en www.b0no.cl",
        "G A N A S T E un premio, entra a bit.ly/ok",
        "Recordatorio reunión lunes 10:00, confirma tu asistencia y retira tu regalo en tinyurl.com/r3g4lo",
        "Felicidades ganaste un iPhone gratis, reclama en bit.ly/iphone",
        "Mamá llego tarde, almorzamos el domingo. Sorteo de $1.000.000 en www.s0rte0.cl",
        "Estimado, su cuenta será bloqueada, verifique en http://banco-seguro.xyz",
        "p.r.e.m.i.o para ti en bit.ly/a1",
    ]

res = pd.DataFrame({"mensaje": ataques})
res["prob_spam"] = modelo.predict_proba(vectorizer.transform(res["mensaje"]))[:, 1]
res["evadio"] = res["prob_spam"] < 0.5
print(f"Intentos: {len(res)} | Evadieron el filtro: {res['evadio'].sum()} ({res['evadio'].mean():.0%})")
res.sort_values("prob_spam")
""")

md("### ¿Por qué se escaparon? Las anomalías")
code("""
for m in res.loc[res["evadio"], "mensaje"]:
    clasificar(m)
    print()
""")
code("""
plt.hist(res["prob_spam"], bins=10, range=(0, 1), color="#2e7d32", edgecolor="white")
plt.axvline(0.5, color="#c62828", ls="--", label="umbral 0,5")
plt.xlabel("P(spam)"); plt.ylabel("mensajes"); plt.title("Distribución de los ataques"); plt.legend(); plt.show()
""")
md("""
Patrones típicos de evasión:
- **Leetspeak** (`pr3m1o`): palabras que el modelo nunca vio, así que las ignora.
- **Separar letras** (`p.r.e.m.i.o`, `G A N A S T E`): se rompen en tokens de una letra, que se descartan.
- **Diluir** con palabras "normales" (profe, informe, mamá, reunión) que empujan hacia normal.
""")

md("""
## 7. Reentrenar con los casos que se escaparon
Agregamos los mensajes que evadieron el filtro, etiquetados como spam, y reentrenamos.

> Ojo: esto **no** demuestra que el modelo aprendió a detectar evasiones en general. Le enseñamos *los casos
> que se escaparon*. Para saber si generaliza, lo probamos con variantes nuevas que no vio.
""")
code("""
escapados = res.loc[res["evadio"], "mensaje"].tolist()
X_aug = pd.concat([X_train, pd.Series(escapados)], ignore_index=True)
y_aug = pd.concat([y_train, pd.Series([1] * len(escapados))], ignore_index=True)

vectorizer2 = TfidfVectorizer(lowercase=True, stop_words=STOPWORDS_ES)
modelo2 = LogisticRegression(max_iter=1000).fit(vectorizer2.fit_transform(X_aug), y_aug)

res["prob_spam_v2"] = modelo2.predict_proba(vectorizer2.transform(res["mensaje"]))[:, 1]
print(f"Exactitud en test original con el modelo v2: "
      f"{accuracy_score(y_test, modelo2.predict(vectorizer2.transform(X_test))):.1%}")
res[["mensaje", "prob_spam", "prob_spam_v2"]].round(3).sort_values("prob_spam")
""")
code("""
variantes_nuevas = [
    "Hola equipo, el acta quedó lista. B0n0 especial en bit.ly/zz",
    "r e g a l o para ti en www.r3g4l0.cl",
    "Gracias por venir al taller, pr3mi0 sorpresa en tinyurl.com/q1",
]
pd.DataFrame({
    "mensaje": variantes_nuevas,
    "v1": modelo.predict_proba(vectorizer.transform(variantes_nuevas))[:, 1],
    "v2": modelo2.predict_proba(vectorizer2.transform(variantes_nuevas))[:, 1],
}).round(3)
""")

md("""
## 8. Defensas y límites
- **Normalizar el texto** antes del modelo (`pr3m1o → premio`, unir letras separadas).
- **Tratar las URL como señal propia** (dominio, acortadores), en vez de partirlas en palabras.
- **Reentrenar con los errores** de forma continua y **monitorear** qué se escapa.
- **Límite de esta demo:** datos simulados y un modelo lineal simple. El atacante siempre puede inventar la
  siguiente variante: es el juego del gato y el ratón.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parent / "charla_spam.ipynb"
nbf.write(nb, out)
print("Escrito", out)
