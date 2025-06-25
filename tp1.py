import numpy as np
import matplotlib.pyplot as plt

GRAVEDAD = 9.81 
MASA = 800
F_MOTOR_MAX = 3000 
F_FRENO_MAX = -3000
A_MAX = 6 * GRAVEDAD
V_INICIAL = 50
DT = 0.01

puntos = {
    'p1': (97.38, 16.6),
    'p2': (78.29, 6.7),
    'p3': (64.31, 5.8),
    'p4': (22.38, 16.8),
    'p5': (19.09, 54.5),
    'p6': (62.04, 77.1)
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

# --- Calcula aceleración lateral ---
def aceleracion_lateral(v, r):
    return v**2 / r if r > 0 else 0


def fuerza_recta(v, v_max_curva):
    if v > v_max_curva:
        return F_FRENO_MAX
    elif v < v_max_curva:
        return F_MOTOR_MAX
    else:
        return 0

def f_recta(y, t, longitud, v_max_curva):
    x, v = y
    F = fuerza_recta(v, v_max_curva)
    a = F / MASA
    if x >= longitud:
        a = 0
    return np.array([v, a])

def f_curva_rk4(y, t):
    theta, omega = y
    return np.array([omega, 0])  # omega constante, sin aceleración angular

def simular_trayectoria():
    d_recta1 = distancia_puntos(puntos['p1'], puntos['p2'])
    ang1 = angulo_arco(centros['arco1'], puntos['p2'], puntos['p3'])
    d_recta2 = distancia_puntos(puntos['p3'], puntos['p4'])
    ang2 = angulo_arco(centros['arco2'], puntos['p4'], puntos['p5'])
    d_recta_final = distancia_puntos(puntos['p5'], puntos['p6'])

    segmentos = [
        ('recta', d_recta1, None),
        ('curva', radios['arco1'], ang1),
        ('recta', d_recta2, None),
        ('curva', radios['arco2'], ang2),
        ('recta', d_recta_final, None)
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
        print(f"\n--- Tramo {i+1}: Tipo = {tipo} ---")
        print(f"Velocidad inicial tramo: {v_actual:.2f} m/s")

        if tipo == 'recta':
            longitud = tramo[1]
            radio_curva_siguiente = tramo[2]


            if radio_curva_siguiente is None:
                v_max_curva = np.inf
            else:
                v_max_curva = np.sqrt(A_MAX * radio_curva_siguiente)

            y0 = np.array([0, v_actual])
            t = np.arange(0, 60, DT)
            res = [y0]
            tiempo_real = [0]

            for j in range(1, len(t)):
                dt = t[j] - t[j-1]
                k1 = f_recta(res[-1], t[j-1], longitud, v_max_curva)
                k2 = f_recta(res[-1] + dt/2 * k1, t[j-1] + dt/2, longitud, v_max_curva)
                k3 = f_recta(res[-1] + dt/2 * k2, t[j-1] + dt/2, longitud, v_max_curva)
                k4 = f_recta(res[-1] + dt * k3, t[j-1] + dt, longitud, v_max_curva)
                y_next = res[-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

                if j % 100 == 0:
                    a_actual = f_recta(res[-1], t[j-1], longitud, v_max_curva)[1]
                    print(f"t={t[j]:.2f}s, pos={y_next[0]:.2f}m, v={y_next[1]:.2f}m/s, a={a_actual:.2f}m/s²")

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

            print(f"Velocidad final tramo recta: {v_actual:.2f} m/s")
            print(f"Tiempo tramo recta: {t_tramo:.2f} s")
            print(f"Distancia tramo recta: {longitud:.2f} m")

            tiempos_totales.extend(tiempo_acum - t_tramo + np.array(tiempo_real))
            posiciones_totales.extend(pos_acum - longitud + res[:,0])
            velocidades_totales.extend(res[:,1])
            fuerzas_totales.extend([
                fuerza_recta(v, v_max_curva) for v in res[:,1]
            ])
            aceleraciones_lat_totales.extend([0]*len(res))

        elif tipo == 'curva':
            radio = tramo[1]
            angulo = tramo[2]

            v_max = np.sqrt(A_MAX * radio)
            v_actual = min(v_actual, v_max)

            t_angular = np.arange(0, angulo / (v_actual / radio) + DT, DT)
            y0 = np.array([0, v_actual / radio])

            res = rk4(f_curva_rk4, y0, t_angular)

            tiempo_acum += t_angular[-1]
            pos_acum += radio * angulo

            print(f"Velocidad tramo curva: {v_actual:.2f} m/s")
            print(f"Tiempo tramo curva: {t_angular[-1]:.2f} s")
            print(f"Longitud arco curva: {radio*angulo:.2f} m")
            print(f"Aceleración lateral máxima permitida: {A_MAX:.2f} m/s²")

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

def graficar(tiempos, posiciones, velocidades, fuerzas, aceleraciones_lat):
    _, axs = plt.subplots(2, 2, figsize=(14, 10))
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


def main():
    tiempos, posiciones, velocidades, fuerzas, aceleraciones_lat = simular_trayectoria()
    print(f"Tiempo total de la trayectoria: {tiempos[-1]:.2f} segundos")
    graficar(tiempos, posiciones, velocidades, fuerzas, aceleraciones_lat)
main()