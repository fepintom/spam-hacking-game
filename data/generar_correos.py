"""Genera data/correos_es.csv: 1.000 correos simulados en español (400 spam, 600 normal).

Los mensajes se arman combinando plantillas y piezas al azar. Se incluyen casos
"difíciles" a propósito (correos normales que hablan de premios o descuentos, y
spam sin palabras obvias) para que el modelo no llegue a un 100% trivial.

Uso:  python data/generar_correos.py
"""
import random
from pathlib import Path

import pandas as pd

SEED = 42
N_SPAM, N_NORMAL = 400, 600
OUT = Path(__file__).parent / "correos_es.csv"

rng = random.Random(SEED)
pick = rng.choice

# ---------------------------------------------------------------- spam
SPAM_APERTURA = [
    "¡Felicidades!", "Estimado cliente,", "URGENTE:", "Hola,", "Atención:", "Último aviso:",
    "Querido usuario,", "¡Enhorabuena!", "Aviso importante:", "", "Oferta exclusiva:",
]
SPAM_GANCHO = [
    "ganaste un premio de ${monto}", "fuiste seleccionado para recibir un bono de ${monto}",
    "tu cuenta será bloqueada en {horas} horas", "tienes un regalo pendiente por retirar",
    "obtén un {pct}% de descuento solo hoy", "ganaste un {producto} gratis",
    "tu préstamo preaprobado de ${monto} está listo", "tienes un reembolso de ${monto} sin cobrar",
    "participas en el sorteo de un {producto}", "recibe dinero extra trabajando desde casa",
    "detectamos un problema con tu tarjeta", "tu paquete está retenido por falta de pago",
    "accede a crédito inmediato sin revisar Dicom", "gana ${monto} diarios con este método",
]
SPAM_ACCION = [
    "Haz clic aquí", "Reclama tu premio", "Ingresa ahora", "Confirma tus datos", "Responde YA",
    "Verifica tu cuenta", "Retíralo hoy", "Activa tu beneficio", "Entra al link",
]
SPAM_LINK = [
    "bit.ly/{cod}", "www.premios-{cod}.com", "http://cl-{cod}.xyz/login", "www.bono{cod}.net",
    "http://tinyurl.com/{cod}", "www.sorteo-chile-{cod}.info", "http://{cod}.link/pago",
]
SPAM_CIERRE = [
    "Oferta válida por tiempo limitado.", "No pierdas esta oportunidad.", "Cupos limitados.",
    "Si no respondes perderás el beneficio.", "100% garantizado.", "", "Sin costo para ti.",
]
# Spam "sutil": sin palabras gancho típicas (casos difíciles)
SPAM_SUTIL = [
    "Hola, te escribo por el pedido pendiente, revisa el detalle en {link}",
    "Tu documento compartido está disponible: {link}",
    "Necesitamos que actualices tu información antes del viernes en {link}",
    "Te dejo la factura adjunta, descárgala aquí {link}",
    "Hay un mensaje nuevo esperando tu respuesta, míralo en {link}",
    "Hola {nombre}, ¿esta foto es tuya? {link}",
    "Revisa el comprobante de la transferencia del {dia}: {link}",
    "Tu reunión del {dia} fue reprogramada, confirma en {link}",
]
PRODUCTOS = ["iPhone", "televisor", "notebook", "viaje a Cancún", "scooter", "PlayStation", "smartwatch"]

# ---------------------------------------------------------------- normal
NORMAL = [
    "Hola {nombre}, ¿nos juntamos el {dia} a las {hora} para revisar el {tema}?",
    "Te envío el informe de {tema} que quedamos en revisar. Cualquier comentario me avisas.",
    "Recordatorio: la reunión de {tema} se movió al {dia} a las {hora}.",
    "{nombre}, ¿me puedes mandar la presentación de {tema} antes del {dia}?",
    "Gracias por tu ayuda con {tema}, quedó muy bien.",
    "Mamá, llego tarde hoy, ¿almorzamos el {dia}?",
    "Hola, ¿alguien tiene los apuntes de la clase de {tema}?",
    "El {dia} no hay clases de {tema}, se recupera la próxima semana.",
    "Confirmo mi asistencia al taller de {tema} del {dia}.",
    "¿Viste el partido ayer? Juguemos fútbol el {dia} en la tarde.",
    "Adjunto la boleta de la compra del {dia} para tu rendición.",
    "Te aviso que el pago del arriendo lo hice el {dia}.",
    "Hola {nombre}, feliz cumpleaños, que lo pases increíble.",
    "La entrega del proyecto de {tema} es el {dia} a las {hora}.",
    "Quedó agendada la hora al dentista para el {dia} a las {hora}.",
    "Oye {nombre}, ¿pasas a buscarme el {dia}? Salgo a las {hora}.",
]
# Normales "difíciles": usan palabras que el spam también usa
NORMAL_DIFICIL = [
    "{nombre}, ganaste el premio al mejor proyecto de {tema}, felicitaciones de todo el equipo.",
    "El supermercado tiene descuento en frutas el {dia}, ¿vamos juntos?",
    "Me llegó el bono de fin de año, el {dia} te pago lo que te debo.",
    "Revisé mi cuenta y el depósito llegó bien, gracias {nombre}.",
    "El sorteo del amigo secreto de la oficina es el {dia} a las {hora}.",
    "Encontré una oferta de pasajes para el {dia}, te mando el link de la aerolínea.",
    "La entrada al museo es gratis el {dia}, ¿te animas?",
    "Hola, te dejo el link de la reunión de {tema}: meet.google.com/{cod}",
    "Urgente {nombre}: necesito la planilla de {tema} antes de las {hora}.",
    "Confirma tu asistencia a la clase de {tema} del {dia}, por favor.",
    "Haz clic en el enlace del aula virtual para ver la nota de {tema}.",
    "Quedan cupos limitados para el taller de {tema} del {dia}, inscríbete con {nombre}.",
]
NOMBRES = ["Camila", "Diego", "Valentina", "Matías", "Javiera", "Felipe", "Sofía", "Tomás", "Fernanda", "Ignacio"]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
TEMAS = ["estadística", "big data", "Python", "ventas", "inventario", "marketing", "presupuesto", "matemáticas"]


def cod():
    return "".join(rng.choice("abcdefghijkmnpqrstuvwxyz23456789") for _ in range(rng.randint(4, 7)))


def fill(s):
    return s.format(
        monto=f"{rng.choice([50, 100, 250, 500, 1000, 2000])}.000",
        horas=rng.choice([12, 24, 48]),
        pct=rng.choice([50, 70, 80, 90]),
        producto=pick(PRODUCTOS),
        cod=cod(),
        nombre=pick(NOMBRES),
        dia=pick(DIAS),
        hora=f"{rng.randint(8, 20)}:{pick(['00', '15', '30', '45'])}",
        tema=pick(TEMAS),
        link=pick(SPAM_LINK).format(cod=cod()),
    )


def spam():
    if rng.random() < 0.2:
        return fill(pick(SPAM_SUTIL))
    partes = [pick(SPAM_APERTURA), fill(pick(SPAM_GANCHO)).capitalize() + ".",
              pick(SPAM_ACCION) + ": " + fill(pick(SPAM_LINK)), pick(SPAM_CIERRE)]
    return " ".join(p for p in partes if p).strip()


def normal():
    return fill(pick(NORMAL_DIFICIL) if rng.random() < 0.25 else pick(NORMAL))


def unicos(gen, n, etiqueta):
    vistos = set()
    while len(vistos) < n:
        vistos.add(gen())
    return [(t, etiqueta) for t in sorted(vistos)]


rows = unicos(spam, N_SPAM, "spam") + unicos(normal, N_NORMAL, "normal")
rng.shuffle(rows)
df = pd.DataFrame(rows, columns=["texto", "etiqueta"])
df.to_csv(OUT, index=False)
print(f"Guardado {OUT} | {len(df)} filas | {df.etiqueta.value_counts().to_dict()} | únicos: {df.texto.nunique()}")
