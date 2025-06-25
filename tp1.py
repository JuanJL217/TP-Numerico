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

def rk4(f, y0, t, *args, **kwargs):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        k1 = f(y[i-1], t[i-1], *args, **kwargs)
        k2 = f(y[i-1] + dt/2 * k1, t[i-1] + dt/2, *args, **kwargs)
        k3 = f(y[i-1] + dt/2 * k2, t[i-1] + dt/2, *args, **kwargs)
        k4 = f(y[i-1] + dt * k3, t[i-1] + dt, *args, **kwargs)
        y[i] = y[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

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
    a = fuerza_recta(v, v_max_curva) / MASA if x < longitud else 0
    return np.array([v, a])

def f_curva(y, t):
    _, omega = y
    return np.array([omega, 0])

def simular_trayectoria():
    d1 = distancia_puntos(puntos['p1'], puntos['p2'])
    ang1 = angulo_arco(centros['arco1'], puntos['p2'], puntos['p3'])
    d2 = distancia_puntos(puntos['p3'], puntos['p4'])
    ang2 = angulo_arco(centros['arco2'], puntos['p4'], puntos['p5'])
    d3 = distancia_puntos(puntos['p5'], puntos['p6'])

    segmentos = [
        ('recta', d1, radios['arco1']),
        ('curva', radios['arco1'], ang1),
        ('recta', d2, radios['arco2']),
        ('curva', radios['arco2'], ang2),
        ('recta', d3, None)
    ]

    tiempos_totales, posiciones_totales = [], []
    velocidades_totales, fuerzas_totales = [], []
    aceleraciones_lat_totales = []

    v_actual = V_INICIAL
    pos_acum = 0
    tiempo_acum = 0

    for i, tramo in enumerate(segmentos):
        tipo, valor1, valor2 = tramo
        print(f"\n--- Tramo {i+1}: Tipo = {tipo} ---")
        print(f"Velocidad inicial: {v_actual:.2f} m/s")

        if tipo == 'recta':
            longitud = valor1
            radio_curva_siguiente = valor2
            v_max_curva = np.sqrt(A_MAX * radio_curva_siguiente) if radio_curva_siguiente else np.inf

            y0 = np.array([0, v_actual])
            t = np.arange(0, 60, DT)
            res = rk4(f_recta, y0, t, longitud, v_max_curva)

            idx_final = np.argmax(res[:,0] >= longitud)
            if res[idx_final, 0] > longitud:
                res[idx_final, 0] = longitud
            res = res[:idx_final+1]
            t = t[:idx_final+1]

            v_actual = res[-1,1]
            pos_acum += longitud
            tiempo_acum += t[-1]

            print(f"Tiempo tramo recta: {t[-1]:.2f} s")

            tiempos_totales.extend(tiempo_acum - t[-1] + t)
            posiciones_totales.extend(pos_acum - longitud + res[:,0])
            velocidades_totales.extend(res[:,1])
            fuerzas_totales.extend([fuerza_recta(v, v_max_curva) for v in res[:,1]])
            aceleraciones_lat_totales.extend([0]*len(res))

        elif tipo == 'curva':
            radio, angulo = valor1, valor2
            v_max = np.sqrt(A_MAX * radio)
            v_actual = min(v_actual, v_max)
            omega = v_actual / radio

            t = np.arange(0, angulo / omega + DT, DT)
            y0 = np.array([0, omega])
            res = rk4(f_curva, y0, t)

            tiempo_acum += t[-1]
            pos_acum += radio * angulo

            print(f"Tiempo tramo curva: {t[-1]:.2f} s")

            tiempos_totales.extend(tiempos_totales[-1] + t if tiempos_totales else t)
            posiciones_totales.extend([pos_acum - radio*angulo + radio * y[0] for y in res])
            velocidades_totales.extend([y[1] * radio for y in res])
            fuerzas_totales.extend([0]*len(res))
            aceleraciones_lat_totales.extend([aceleracion_lateral(y[1]*radio, radio) for y in res])

    return np.array(tiempos_totales)


def main():
    tiempos = simular_trayectoria()
    print(f"\nTiempo total de la trayectoria: {tiempos[-1]:.2f} segundos")
main()