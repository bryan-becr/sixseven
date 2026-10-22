"""
SIX SEVEN - Semana de Ciencias
------------------------------
Juego interactivo: la camara detecta el gesto "six seven" (las dos manos
subiendo y bajando de forma ALTERNADA) y cuenta cuantos haces en una ronda.

Flujo:  Pantalla de inicio -> cuenta regresiva 3,2,1 -> ronda de 20 s
        -> resultados (con record guardado) -> otra vez.

Como funciona (para explicarlo en tu stand):
1. MediaPipe detecta 21 puntos clave en cada mano.
2. Tomamos la altura (eje Y) de la muneca de cada mano.
3. Calculamos la diferencia de altura entre las dos manos.
4. Si esa diferencia cambia de signo varias veces en poco tiempo
   (mano izquierda arriba -> derecha arriba -> izquierda arriba...),
   es un movimiento alternado = "six seven".
5. Todo se mide relativo al tamano de la mano, asi funciona
   cerca o lejos de la camara.

Instalacion:
    pip install opencv-python mediapipe numpy

Uso:
    python sixseven.py                  (camara 0)
    python sixseven.py 1                (otra camara, por numero)
    python sixseven.py http://IP:4747/video   (camara por red)

Teclas:
    ESPACIO / ENTER = empezar        q o ESC = salir
    f = pantalla completa            d = mostrar/ocultar datos tecnicos
    r = borrar el record (en la pantalla de inicio)
"""

import math
import os
import sys
import threading
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


CAMERA_INDEX = 0          # prueba 1 si tienes otra camara
VENTANA_SEG = 1.5         # ventana de tiempo para contar alternancias
CAMBIOS_NECESARIOS = 2    # cambios de lado necesarios para validar el gesto
AMPLITUD_REL = 0.2        # diferencia minima de altura (x tamano de mano)
ENFRIAMIENTO_SEG = 0.2    # pausa tras detectar, para no contar doble
DURACION_MENSAJE = 0.4    # cuanto tiempo se muestra "SIX SEVEN!"

DURACION_RONDA = 20       # segundos que dura cada ronda
CUENTA_REGRESIVA = 2      # segundos de cuenta regresiva antes de empezar
AUTORETORNO_SEG = 10      # tras los resultados, vuelve solo al inicio
# --------------------------------------------------------

VENTANA = "Six Seven - Semana de Ciencias"
CARPETA = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CARPETA, "hand_landmarker.task")
RECORD_PATH = os.path.join(CARPETA, "record.txt")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


NARANJA = (0, 140, 255)
BLANCO = (255, 255, 255)
AMARILLO = (0, 215, 255)
VERDE = (80, 220, 80)
GRIS = (30, 30, 30)

# Conexiones entre los 21 puntos de la mano (para dibujar el esqueleto)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),          # indice
    (5, 9), (9, 10), (10, 11), (11, 12),     # medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # anular
    (13, 17), (17, 18), (18, 19), (19, 20),  # menique
    (0, 17),                                  # palma
]


# ======================= Camara =======================
class LectorCamara:
    """Lee la camara en un hilo aparte y se queda siempre con el frame MAS
    NUEVO. Asi no se acumulan imagenes viejas (el retraso tipico de camaras
    por Wi-Fi) y el programa no espera a la camara."""

    def __init__(self, fuente):
        self.cap = cv2.VideoCapture(fuente)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.lock = threading.Lock()
        self.frame = None
        self.nuevo = False
        self.activo = self.cap.isOpened()
        if self.activo:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.activo:
            ok, f = self.cap.read()
            if not ok:
                self.activo = False
                break
            with self.lock:
                self.frame = f
                self.nuevo = True

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        t0 = time.time()
        while self.activo and time.time() - t0 < 2.0:
            with self.lock:
                if self.nuevo:
                    self.nuevo = False
                    return True, self.frame.copy()
            time.sleep(0.003)
        return False, None

    def release(self):
        self.activo = False
        time.sleep(0.05)
        self.cap.release()


# ======================= Utilidades =======================
def asegurar_modelo():
    if os.path.exists(MODEL_PATH):
        return
    print("Descargando modelo de manos (solo la primera vez)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise SystemExit(
            f"No se pudo descargar el modelo: {e}\n"
            f"Descargalo manualmente de:\n{MODEL_URL}\n"
            f"y guardalo como 'hand_landmarker.task' junto a este archivo."
        )
    print("Modelo descargado.")


def cargar_record():
    try:
        with open(RECORD_PATH) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def guardar_record(valor):
    try:
        with open(RECORD_PATH, "w") as f:
            f.write(str(int(valor)))
    except Exception:
        pass


def tamano_mano(lm):
    """Distancia muneca -> base del dedo medio (referencia de escala)."""
    w, m = lm[0], lm[9]
    return ((w.x - m.x) ** 2 + (w.y - m.y) ** 2) ** 0.5


def dibujar_mano(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def oscurecer(frame, cantidad):
    """Oscurece la imagen (0 = igual, 1 = negro)."""
    return cv2.convertScaleAbs(frame, alpha=1.0 - cantidad, beta=0)


def texto(frame, txt, y, escala, color=BLANCO, grosor=2):
    """Texto centrado horizontalmente, con sombra para que se lea siempre."""
    fuente = cv2.FONT_HERSHEY_DUPLEX
    (tw, _), _ = cv2.getTextSize(txt, fuente, escala, grosor)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, txt, (x + 3, y + 3), fuente, escala, (0, 0, 0), grosor + 3)
    cv2.putText(frame, txt, (x, y), fuente, escala, color, grosor)


# ======================= Pantallas =======================
def pantalla_inicio(frame, record, ahora):
    h, w = frame.shape[:2]
    k = w / 640.0
    frame = oscurecer(frame, 0.6)
    texto(frame, "SIX SEVEN", int(h * 0.26), 2.8 * k, NARANJA, int(6 * k))
    texto(frame, "Semana de Ciencias", int(h * 0.36), 1.0 * k, BLANCO, max(1, int(2 * k)))
    texto(frame, "Pon las dos manos frente a la camara", int(h * 0.50), 0.8 * k)
    texto(frame, "y subelas y bajalas: una arriba, otra abajo", int(h * 0.58), 0.8 * k)
    texto(frame, f"Tienes {DURACION_RONDA} segundos. Cuantos puedes hacer?",
          int(h * 0.66), 0.8 * k, AMARILLO)
    if record > 0:
        texto(frame, f"Record: {record}", int(h * 0.76), 1.0 * k, VERDE)
    if int(ahora * 2) % 2 == 0:  # parpadeo
        texto(frame, "Pulsa ESPACIO para empezar", int(h * 0.90), 1.0 * k, BLANCO)
    return frame


def pantalla_cuenta(frame, restante):
    h, w = frame.shape[:2]
    k = w / 640.0
    frame = oscurecer(frame, 0.35)
    texto(frame, "Prepara las manos!", int(h * 0.20), 1.3 * k, AMARILLO)
    numero = str(max(1, math.ceil(restante)))
    texto(frame, numero, int(h * 0.68), 9.0 * k, NARANJA, int(18 * k))
    return frame


def hud_juego(frame, puntos, restante, mostrar_banner, dos_manos):
    h, w = frame.shape[:2]
    k = w / 640.0
    barra = int(64 * k)
    cv2.rectangle(frame, (0, 0), (w, barra), GRIS, -1)
    cv2.putText(frame, f"Six Seven: {puntos}", (int(15 * k), int(46 * k)),
                cv2.FONT_HERSHEY_DUPLEX, 1.3 * k, BLANCO, max(1, int(2 * k)))
    t_txt = f"{max(0, math.ceil(restante))} s"
    (tw, _), _ = cv2.getTextSize(t_txt, cv2.FONT_HERSHEY_DUPLEX, 1.3 * k, max(1, int(2 * k)))
    color_t = (60, 60, 255) if restante <= 5 else BLANCO
    cv2.putText(frame, t_txt, (w - tw - int(15 * k), int(46 * k)),
                cv2.FONT_HERSHEY_DUPLEX, 1.3 * k, color_t, max(1, int(2 * k)))

    # barra de tiempo
    frac = max(0.0, min(1.0, restante / DURACION_RONDA))
    cv2.rectangle(frame, (0, barra), (w, barra + int(10 * k)), (70, 70, 70), -1)
    cv2.rectangle(frame, (0, barra), (int(w * frac), barra + int(10 * k)),
                  (60, 60, 255) if restante <= 5 else NARANJA, -1)

    if mostrar_banner:
        y0 = barra + int(20 * k)
        cv2.rectangle(frame, (0, y0), (w, y0 + int(70 * k)), NARANJA, -1)
        texto(frame, "SIX SEVEN!", y0 + int(54 * k), 1.8 * k, BLANCO, max(2, int(4 * k)))

    if not dos_manos:
        texto(frame, "Muestra las dos manos a la camara", h - int(20 * k), 0.8 * k, AMARILLO)
    return frame


def pantalla_resultado(frame, puntos, record, nuevo_record, ahora):
    h, w = frame.shape[:2]
    k = w / 640.0
    frame = oscurecer(frame, 0.6)
    texto(frame, "TIEMPO!", int(h * 0.20), 2.2 * k, BLANCO, int(5 * k))
    texto(frame, str(puntos), int(h * 0.55), 6.0 * k, NARANJA, int(12 * k))
    texto(frame, "six seven" if puntos == 1 else "six sevens", int(h * 0.65), 1.2 * k)
    if nuevo_record:
        if int(ahora * 3) % 2 == 0:
            texto(frame, "NUEVO RECORD!", int(h * 0.76), 1.6 * k, VERDE, int(3 * k))
    else:
        texto(frame, f"Record: {record}", int(h * 0.76), 1.0 * k, AMARILLO)
    texto(frame, "ESPACIO: jugar otra vez", int(h * 0.90), 0.9 * k)
    return frame


# ======================= Programa principal =======================
def main():
    asegurar_modelo()

    # Fuente de video: python sixseven.py [numero_de_camara | URL]
    fuente = CAMERA_INDEX
    if len(sys.argv) > 1:
        fuente = int(sys.argv[1]) if sys.argv[1].isdigit() else sys.argv[1]

    cap = LectorCamara(fuente)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara/fuente: {fuente}")
        print("Prueba con otro numero (0, 1, 2...) o revisa la URL.")
        return

    opciones = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    pantalla_completa = False

    estado = "inicio"            # inicio | cuenta | juego | resultado
    t_estado = time.time()
    puntos = 0
    record = cargar_record()
    nuevo_record = False

    lado_actual = 0              # -1 / +1: que mano esta mas arriba
    cambios = deque()            # tiempos de cada cambio de lado
    ultimo_trigger = 0.0
    mostrar_hasta = 0.0
    mostrar_datos = False
    t_prev = time.time()
    t_inicio = time.time()
    ultimo_ts_ms = -1

    def iniciar_cuenta(ahora):
        nonlocal estado, t_estado
        estado = "cuenta"
        t_estado = ahora

    with vision.HandLandmarker.create_from_options(opciones) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # efecto espejo
            ahora = time.time()

            # MediaPipe exige timestamps que siempre crezcan
            ts_ms = int((ahora - t_inicio) * 1000)
            if ts_ms <= ultimo_ts_ms:
                ts_ms = ultimo_ts_ms + 1
            ultimo_ts_ms = ts_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect_for_video(mp_image, ts_ms)

            manos_det = res.hand_landmarks or []
            diff_rel = 0.0
            dos_manos = len(manos_det) == 2

            # ---------- Deteccion del movimiento alternado ----------
            if dos_manos:
                # Ordenamos por posicion horizontal: izquierda y derecha de la imagen
                izq, der = sorted(manos_det, key=lambda l: l[0].x)
                escala = max((tamano_mano(izq) + tamano_mano(der)) / 2, 1e-3)
                diff_rel = (der[0].y - izq[0].y) / escala   # y crece hacia abajo

            if estado == "juego":
                if dos_manos:
                    lado = 0
                    if diff_rel > AMPLITUD_REL:
                        lado = 1
                    elif diff_rel < -AMPLITUD_REL:
                        lado = -1
                    if lado != 0:
                        if lado_actual != 0 and lado != lado_actual:
                            cambios.append(ahora)
                        lado_actual = lado
                elif cambios and ahora - cambios[-1] > VENTANA_SEG:
                    lado_actual = 0

                while cambios and ahora - cambios[0] > VENTANA_SEG:
                    cambios.popleft()

                if (
                    len(cambios) >= CAMBIOS_NECESARIOS
                    and ahora - ultimo_trigger > ENFRIAMIENTO_SEG
                ):
                    puntos += 1
                    ultimo_trigger = ahora
                    mostrar_hasta = ahora + DURACION_MENSAJE
                    cambios.clear()  # se conserva lado_actual: el seguimiento no se interrumpe
            else:
                cambios.clear()
                lado_actual = 0

            # ---------- Cambios de estado ----------
            if estado == "cuenta" and ahora - t_estado >= CUENTA_REGRESIVA:
                estado = "juego"
                t_estado = ahora
                puntos = 0
                ultimo_trigger = 0.0
                mostrar_hasta = 0.0
            elif estado == "juego" and ahora - t_estado >= DURACION_RONDA:
                estado = "resultado"
                t_estado = ahora
                nuevo_record = puntos > record
                if nuevo_record:
                    record = puntos
                    guardar_record(record)
            elif estado == "resultado" and ahora - t_estado >= AUTORETORNO_SEG:
                estado = "inicio"
                t_estado = ahora

            # ---------- Dibujo ----------
            if estado == "inicio":
                frame = pantalla_inicio(frame, record, ahora)
            elif estado == "cuenta":
                frame = pantalla_cuenta(frame, CUENTA_REGRESIVA - (ahora - t_estado))
            elif estado == "resultado":
                frame = pantalla_resultado(frame, puntos, record, nuevo_record, ahora)

            # El esqueleto se dibuja encima para que la gente vea que la detecta
            if estado in ("juego", "cuenta"):
                for lm in manos_det:
                    dibujar_mano(frame, lm)

            if estado == "juego":
                frame = hud_juego(frame, puntos, DURACION_RONDA - (ahora - t_estado),
                                  ahora < mostrar_hasta, dos_manos)

            if mostrar_datos:
                fps = 1.0 / max(ahora - t_prev, 1e-6)
                info = (f"{estado} | dif: {diff_rel:+.2f} | "
                        f"cambios: {len(cambios)}/{CAMBIOS_NECESARIOS} | {fps:.0f} fps")
                cv2.putText(frame, info, (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
            t_prev = ahora

            cv2.imshow(VENTANA, frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            elif tecla in (32, 13) and estado in ("inicio", "resultado"):
                iniciar_cuenta(ahora)
            elif tecla == ord("r") and estado == "inicio":
                record = 0
                guardar_record(0)
            elif tecla == ord("d"):
                mostrar_datos = not mostrar_datos
            elif tecla == ord("f"):
                pantalla_completa = not pantalla_completa
                cv2.setWindowProperty(
                    VENTANA, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if pantalla_completa else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()