import numpy as np
import matplotlib.pyplot as plt

# ------------------- CONSTANTES Y FUNCIONES -------------------
GRAVEDAD = 9.81
MASA = 800
ACELERACION_MAX = 6 * GRAVEDAD
DIFERENCIAL_TIEMPO = 0.01

def rk4_system(f, y0, t0, tf, dt):
    n = int((tf - t0) / dt)
    t = np.linspace(t0, tf, n)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        k1 = dt * f(t[i], y[i])
        k2 = dt * f(t[i] + dt/2, y[i] + k1/2)
        k3 = dt * f(t[i] + dt/2, y[i] + k2/2)
        k4 = dt * f(t[i] + dt, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t, y

def curva_edo(t, y, r):
    theta, omega = y
    return np.array([omega, -GRAVEDAD / r * np.sin(theta)])

def recta_edo(t, y, fuerza):
    x, v = y
    a = fuerza(t) / MASA
    return np.array([v, a])

def fuerza_motor(t):
    return 2000

def fuerza_freno(t):
    return -1500

# ------------------- FUNCIÓN PRINCIPAL -------------------
def main():
    # Tramo 1: recta con aceleración
    x0 = [0, 20]
    t1, y1 = rk4_system(lambda t, y: recta_edo(t, y, fuerza_motor), x0, 0, 3, DIFERENCIAL_TIEMPO)

    # Tramo 2: curva
    r = 50
    theta0 = [0, y1[-1, 1]/r]
    t2, y2 = rk4_system(lambda t, y: curva_edo(t, y, r), theta0, t1[-1], t1[-1] + 2, DIFERENCIAL_TIEMPO)
    x2 = r * np.sin(y2[:, 0]) + y1[-1, 0]
    y2_pos = r * (1 - np.cos(y2[:, 0]))

    # Tramo 3: recta con frenado
    x0_3 = [x2[-1], r * y2[-1, 1]]
    t3, y3 = rk4_system(lambda t, y: recta_edo(t, y, fuerza_freno), x0_3, t2[-1], t2[-1] + 3, DIFERENCIAL_TIEMPO)

    # Combinar resultados
    t_total = np.concatenate([t1, t2, t3])
    x_total = np.concatenate([y1[:, 0], x2, y3[:, 0]])
    v_total = np.concatenate([y1[:, 1], r * y2[:, 1], y3[:, 1]])
    a_total = np.gradient(v_total, DIFERENCIAL_TIEMPO)

    # Graficar
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(t_total, x_total)
    axs[0].set_ylabel("Posición (m)")
    axs[0].set_title("Trayectoria de Franco - Posición, Velocidad y Aceleración")

    axs[1].plot(t_total, v_total)
    axs[1].set_ylabel("Velocidad (m/s)")

    axs[2].plot(t_total, a_total)
    axs[2].axhline(ACELERACION_MAX, color='r', linestyle='--', label='6g límite')
    axs[2].axhline(-ACELERACION_MAX, color='r', linestyle='--')
    axs[2].set_ylabel("Aceleración (m/s²)")
    axs[2].set_xlabel("Tiempo (s)")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

main()