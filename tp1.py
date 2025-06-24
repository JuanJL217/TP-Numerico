import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# CONSTANTES Y PARÁMETROS
# -------------------------------
G = 9.81                # gravedad (m/s²)
M = 800                 # masa auto + piloto (kg)

F_MOTOR = 3000          # fuerza de motor en rectas (N)
F_FRENO = -2000         # fuerza de frenado (N)

V_INICIAL = 40          # velocidad inicial (m/s) ~ 144 km/h

RADIO_CURVA1 = 25       # radio curva 1 (m)
RADIO_CURVA2 = 20       # radio curva 2 (m)

TIEMPO_RECTA1 = 2       # duración recta1 (s)
TIEMPO_FRENADO = 1      # duración frenado (s)
TIEMPO_CURVA1 = 2       # duración curva 1 (s)
TIEMPO_RECTA2 = 3       # duración recta 2 (s)
TIEMPO_CURVA2 = 3       # duración curva 2 (s)
TIEMPO_RECTA_FINAL = 2  # duración recta final (s)

# -------------------------------
# Método Runge-Kutta 4 genérico
# -------------------------------
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

# -------------------------------
# EDO para curva (tipo péndulo)
# -------------------------------
def curva_ode(g=G, r=25):
    def f(y, t):
        theta, omega = y
        dtheta_dt = omega
        domega_dt = - (g / r) * np.sin(theta)
        return np.array([dtheta_dt, domega_dt])
    return f

# -------------------------------
# EDO para recta (aceleración F/m)
# -------------------------------
def recta_ode(F=F_MOTOR, m=M):
    a = F / m
    def f(y, t):
        x, v = y
        dx_dt = v
        dv_dt = a
        return np.array([dx_dt, dv_dt])
    return f

# -------------------------------
# Simulación completa de sectores
# -------------------------------
def simular_trayectoria():

    t_recta1 = np.linspace(0, TIEMPO_RECTA1, 100)
    t_frenado = np.linspace(0, TIEMPO_FRENADO, 100)
    t_curva1 = np.linspace(0, TIEMPO_CURVA1, 100)
    t_recta2 = np.linspace(0, TIEMPO_RECTA2, 100)
    t_curva2 = np.linspace(0, TIEMPO_CURVA2, 100)
    t_recta_final = np.linspace(0, TIEMPO_RECTA_FINAL, 100)

    # --------- RECTA 1: aceleración inicial ---------
    y0_recta = [0, V_INICIAL]  # x0 = 0, velocidad inicial
    recta1 = rk4(recta_ode(F=F_MOTOR, m=M), y0_recta, t_recta1)
    v_fin_recta1 = recta1[-1, 1]

    # --------- CÁLCULO VELOCIDAD MÁX CURVA 1 ---------
    v_max_curva1 = np.sqrt(6 * G * RADIO_CURVA1)

    # Si es necesario frenar antes de la curva:
    if v_fin_recta1 > v_max_curva1:
        print(f"⚠️ Velocidad {v_fin_recta1:.2f} m/s > {v_max_curva1:.2f} m/s — Se frena antes de la curva.")
        y0_freno = [0, v_fin_recta1]
        frenado = rk4(recta_ode(F=F_FRENO, m=M), y0_freno, t_frenado)
        v_entrada_curva1 = frenado[-1, 1]
    else:
        frenado = None
        v_entrada_curva1 = v_fin_recta1

    # --------- CURVA 1 ---------
    omega0 = v_entrada_curva1 / RADIO_CURVA1
    curva1 = rk4(curva_ode(g=G, r=RADIO_CURVA1), [0, omega0], t_curva1)

    # --------- RECTA 2 ---------
    y0_recta2 = [0, v_entrada_curva1]
    recta2 = rk4(recta_ode(F=0, m=M), y0_recta2, t_recta2)

    # --------- CURVA 2 ---------
    v_max_curva2 = np.sqrt(6 * G * RADIO_CURVA2)
    v_entrada_curva2 = recta2[-1, 1]

    if v_entrada_curva2 > v_max_curva2:
        print(f"⚠️ Velocidad {v_entrada_curva2:.2f} m/s > {v_max_curva2:.2f} m/s — Se frena antes de curva 2.")
        y0_freno2 = [0, v_entrada_curva2]
        frenado2 = rk4(recta_ode(F=F_FRENO, m=M), y0_freno2, t_frenado)
        v_entrada_curva2 = frenado2[-1, 1]
    else:
        frenado2 = None

    omega0_2 = v_entrada_curva2 / RADIO_CURVA2
    curva2 = rk4(curva_ode(g=G, r=RADIO_CURVA2), [0, omega0_2], t_curva2)

    # --------- RECTA FINAL ---------
    y0_final = [0, v_entrada_curva2]
    recta_final = rk4(recta_ode(F=F_MOTOR, m=M), y0_final, t_recta_final)

    # Construir resultado, incluyendo frenado si fue necesario
    resultado = {
        'recta1': (t_recta1, recta1),
    }
    if frenado is not None:
        resultado['frenado'] = (t_frenado, frenado)
    resultado.update({
        'curva1': (t_curva1, curva1),
        'recta2': (t_recta2, recta2),
        'curva2': (t_curva2, curva2),
        'final': (t_recta_final, recta_final)
    })
    if frenado2 is not None:
        resultado['frenado2'] = (t_frenado, frenado2)

    # Calcular tiempo total
    tiempo_total = 0
    for nombre, (t, _) in resultado.items():
        tiempo_tramo = t[-1] - t[0]
        print(f"⏱️  Tramo {nombre}: {tiempo_tramo:.2f} s")
        tiempo_total += tiempo_tramo

    print(f"\n⏱️🟢 Tiempo total de la trayectoria: {tiempo_total:.2f} segundos\n")

    return resultado

# -------------------------------
# Gráfico de resultados
# -------------------------------
def graficar(resultados):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    tiempo_total = []
    posicion = []
    velocidad = []
    aceleracion = []

    tiempo_acumulado = 0
    for tramo in resultados.values():
        t, datos = tramo
        t_absoluto = t + tiempo_acumulado
        tiempo_total.extend(t_absoluto)
        tiempo_acumulado = t_absoluto[-1]

        if datos.shape[1] == 2:
            x, v = datos[:, 0], datos[:, 1]
            posicion.extend(x)
            velocidad.extend(v)
            aceleracion.extend(np.gradient(v, t[1] - t[0]))
        else:
            posicion.extend(np.zeros_like(t))
            velocidad.extend(np.zeros_like(t))
            aceleracion.extend(np.zeros_like(t))

    axs[0].plot(tiempo_total, posicion, label="Posición")
    axs[0].set_ylabel("Posición (m)")
    axs[0].grid()

    axs[1].plot(tiempo_total, velocidad, label="Velocidad", color="orange")
    axs[1].set_ylabel("Velocidad (m/s)")
    axs[1].grid()

    axs[2].plot(tiempo_total, aceleracion, label="Aceleración", color="green")
    axs[2].axhline(6 * G, linestyle='--', color='r', label="±6g")
    axs[2].axhline(-6 * G, linestyle='--', color='r')
    axs[2].set_ylabel("Aceleración (m/s²)")
    axs[2].set_xlabel("Tiempo (s)")
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    plt.show()

# -------------------------------
# Programa principal
# -------------------------------
if __name__ == "__main__":
    resultados = simular_trayectoria()
    graficar(resultados)
