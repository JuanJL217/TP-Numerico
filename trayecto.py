import numpy as np
import matplotlib.pyplot as plt

# --- Constantes ---
G = 9.81                 # gravedad m/s^2
M = 800                  # masa (kg)
F_MOTOR_MAX = 3000       # fuerza máxima motor (N)
F_FRENO_MAX = -3000      # fuerza máxima frenado (N)
A_MAX = 6 * G            # aceleración lateral máxima (m/s^2)
V_INICIAL = 40           # velocidad inicial m/s (~144 km/h)
DT = 0.01                # paso temporal RK4 (s)

# --- Puntos y geometría ---
puntos = {
    'inicio': (97.38, 16.6),
    'fin_recta1': (78.29, 6.7),
    'fin_arco1': (64.31, 5.8),
    'fin_recta2': (22.38, 16.8),
    'fin_arco2': (19.09, 54.5),
    'final': (62.04, 77.1)
}

centros = {
    'arco1': (70, 30),
    'arco2': (27.69, 36.1)
}

radios = {
    'arco1': 25,
    'arco2': 20
}

def distancia_puntos(p1, p2):
    return np.linalg.norm(np.array(p2) - np.array(p1))

def angulo_arco(centro, p_inicio, p_fin):
    v1 = np.array(p_inicio) - np.array(centro)
    v2 = np.array(p_fin) - np.array(centro)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

# --- RK4 genérico ---
def rk4(f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        k1 = f(y[i-1], t[i-1])
        k2 = f(y[i-1] + dt/2 * k1, t[i-1] + dt/2)
        k3 = f(y[i-1] + dt/2 * k2, t[i-1] + dt/2)
        k4 = f(y[i-1] + dt * k3, t[i-1] + dt)
        y[i] = y[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

# --- EDO para recta ---
# y = [posicion, velocidad]
def f_recta(fuerza):
    def f(y, t):
        x, v = y
        a = fuerza / M
        return np.array([v, a])
    return f

# --- EDO para curva (ángulo theta, velocidad angular omega) ---
# y = [theta, omega]
def f_curva(radio):
    def f(y, t):
        theta, omega = y
        # omega constante, sin aceleración angular
        return np.array([omega, 0])
    return f

# --- Calcula aceleración lateral ---
def aceleracion_lateral(v, r):
    return v**2 / r if r > 0 else 0

# --- Simulación general con control de velocidad y frenado ---
def simular_trayectoria():
    # Calcular distancias y ángulos
    d_recta1 = distancia_puntos(puntos['inicio'], puntos['fin_recta1'])
    ang1 = angulo_arco(centros['arco1'], puntos['fin_recta1'], puntos['fin_arco1'])
    d_recta2 = distancia_puntos(puntos['fin_arco1'], puntos['fin_recta2'])
    ang2 = angulo_arco(centros['arco2'], puntos['fin_recta2'], puntos['fin_arco2'])
    d_recta_final = distancia_puntos(puntos['fin_arco2'], puntos['final'])

    segmentos = [
        ('recta', d_recta1, None),
        ('curva', radios['arco1'], ang1),
        ('recta', d_recta2, radios['arco1']),
        ('curva', radios['arco2'], ang2),
        ('recta', d_recta_final, radios['arco2'])
    ]

    tiempos_totales = []
    posiciones_totales = []
    velocidades_totales = []
    fuerzas_totales = []
    aceleraciones_lat_totales = []

    v_actual = V_INICIAL
    pos_acum = 0
    tiempo_acum = 0

    for i, tramo in enumerate(segmentos):
        tipo = tramo[0]

        if tipo == 'recta':
            longitud = tramo[1]
            radio_curva_siguiente = tramo[2]  # Puede ser None si no hay curva siguiente

            # Si hay curva siguiente, calculo velocidad máxima admisible para no exceder aceleración lateral
            if radio_curva_siguiente is not None:
                v_max_curva = np.sqrt(A_MAX * radio_curva_siguiente)
            else:
                v_max_curva = 70  # velocidad máxima arbitraria si no hay curva

            # Si velocidad actual > v_max_curva, frenamos
            # Defino función fuerza dinámica según necesidad: motor, frenado o nada
            def f(y, t):
                x, v = y
                # Fuerza por defecto 0
                F = 0
                # Si nos pasamos de velocidad máxima curva, frenamos
                if v > v_max_curva:
                    F = F_FRENO_MAX
                # Si velocidad menor al máximo permitido por motor, aceleramos
                elif v < v_max_curva and v < 70:
                    F = F_MOTOR_MAX
                a = F / M
                # Cortar cuando se llega a la distancia de la recta
                if x >= longitud:
                    a = 0
                    v = v  # mantener velocidad
                return np.array([v, a])

            y0 = np.array([0, v_actual])
            t = np.arange(0, 60, DT)  # tiempo máximo para recorrer el tramo

            res = [y0]
            tiempo_real = [0]

            for j in range(1, len(t)):
                dt = t[j] - t[j-1]
                k1 = f(res[-1], t[j-1])
                k2 = f(res[-1] + dt/2 * k1, t[j-1] + dt/2)
                k3 = f(res[-1] + dt/2 * k2, t[j-1] + dt/2)
                k4 = f(res[-1] + dt * k3, t[j-1] + dt)
                y_next = res[-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

                if y_next[0] >= longitud:
                    y_next[0] = longitud
                    res.append(y_next)
                    tiempo_real.append(t[j])
                    break
                res.append(y_next)
                tiempo_real.append(t[j])

            res = np.array(res)
            t_tramo = tiempo_real[-1]
            v_actual = res[-1,1]
            pos_acum += longitud
            tiempo_acum += t_tramo

            tiempos_totales.extend(tiempo_acum - t_tramo + np.array(tiempo_real))
            posiciones_totales.extend(pos_acum - longitud + res[:,0])
            velocidades_totales.extend(res[:,1])
            # Fuerza según regla: motor si acelera, freno si desacelera, 0 si constante
            fuerzas_totales.extend([
                F_MOTOR_MAX if v < v_max_curva and v < 70 else
                F_FRENO_MAX if v > v_max_curva else 0 for v in res[:,1]
            ])
            aceleraciones_lat_totales.extend([0]*len(res))

        elif tipo == 'curva':
            radio = tramo[1]
            angulo = tramo[2]

            # Limitar velocidad para curva
            v_max = np.sqrt(A_MAX * radio)
            v_actual = min(v_actual, v_max)

            t_angular = np.arange(0, angulo / (v_actual / radio) + DT, DT)
            y0 = np.array([0, v_actual / radio])  # theta=0, omega=v/radio

            def f_curva_rk4(y, t_):
                theta, omega = y
                return np.array([omega, 0])  # omega constante

            res = rk4(f_curva_rk4, y0, t_angular)

            tiempo_acum += t_angular[-1]
            pos_acum += radio * angulo

            tiempos_totales.extend(tiempos_totales[-1] + t_angular if tiempos_totales else t_angular)
            posiciones_totales.extend([pos_acum - radio*angulo + radio*res[i,0] for i in range(len(res))])
            velocidades_totales.extend([omega*radio for _, omega in res])
            fuerzas_totales.extend([0]*len(res))
            aceleraciones_lat_totales.extend([aceleracion_lateral(omega*radio, radio) for _, omega in res])

        else:
            raise ValueError("Tipo de tramo desconocido")

    return (np.array(tiempos_totales), np.array(posiciones_totales),
            np.array(velocidades_totales), np.array(fuerzas_totales),
            np.array(aceleraciones_lat_totales))

# --- Graficar resultados ---
def graficar(tiempos, posiciones, velocidades, fuerzas, aceleraciones_lat):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    axs[0].plot(tiempos, posiciones)
    axs[0].set_title("Posición total (m)")
    axs[0].set_xlabel("Tiempo (s)")
    axs[0].set_ylabel("Posición (m)")
    axs[0].grid()

    axs[1].plot(tiempos, velocidades)
    axs[1].set_title("Velocidad (m/s)")
    axs[1].set_xlabel("Tiempo (s)")
    axs[1].set_ylabel("Velocidad (m/s)")
    axs[1].grid()

    axs[2].plot(tiempos, fuerzas)
    axs[2].set_title("Fuerza aplicada (N)")
    axs[2].set_xlabel("Tiempo (s)")
    axs[2].set_ylabel("Fuerza (N)")
    axs[2].grid()

    axs[3].plot(tiempos, aceleraciones_lat)
    axs[3].axhline(A_MAX, color='r', linestyle='--', label='6g límite')
    axs[3].set_title("Aceleración lateral (m/s²)")
    axs[3].set_xlabel("Tiempo (s)")
    axs[3].set_ylabel("Aceleración lateral (m/s²)")
    axs[3].legend()
    axs[3].grid()

    plt.tight_layout()
    plt.show()

# --- Programa principal ---
if __name__ == "__main__":
    tiempos, posiciones, velocidades, fuerzas, aceleraciones_lat = simular_trayectoria()
    print(f"Tiempo total de la trayectoria: {tiempos[-1]:.2f} segundos")
    graficar(tiempos, posiciones, velocidades, fuerzas, aceleraciones_lat)
