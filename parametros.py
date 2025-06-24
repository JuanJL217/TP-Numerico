import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. Configuración ---
ARCHIVO = 'Figure_1.png'  # Cambia por tu imagen
UMBRAL_RECTA = 0.98      # Umbral de correlación para detectar rectas (0 a 1)
RADIO_CURVA = 30         # Radio fijo para curvas (metros)

# --- 2. Procesamiento de imagen ---
img = cv2.imread(ARCHIVO)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)  # Ajusta 250 para eliminar cuadrícula

# --- 3. Extraer curva principal ---
contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
curva = max(contornos, key=cv2.contourArea).squeeze()

# --- 4. Convertir a coordenadas reales ---
ESCALA_X, ESCALA_Y = 0.1, 0.1  # Ajusta según tu gráfico
x = curva[:, 0] * ESCALA_X
y = (img.shape[0] - curva[:, 1]) * ESCALA_Y  # Invertir Y

# --- 5. Detección de segmentos ---
segmentos = []
inicio = 0
tolerancia = 10  # Puntos mínimos para analizar

for fin in range(tolerancia, len(x), 5):  # Saltar cada 5 puntos para eficiencia
    # Verificar si es recta
    X = x[inicio:fin].reshape(-1, 1)
    Y = y[inicio:fin]
    modelo = LinearRegression().fit(X, Y)
    score = modelo.score(X, Y)  # Coeficiente R²
    
    if score > UMBRAL_RECTA:  # Es recta
        angulo = np.degrees(np.arctan(modelo.coef_[0]))
        longitud = np.sqrt((x[fin] - x[inicio])**2 + (y[fin] - y[inicio])**2)
        segmentos.append({
            'tipo': 'recta',
            'inicio': (x[inicio], y[inicio]),
            'fin': (x[fin], y[fin]),
            'longitud': longitud,
            'angulo': angulo
        })
        inicio = fin
    else:  # Es curva
        longitud_arco = np.sum(np.sqrt(np.diff(x[inicio:fin])**2 + np.diff(y[inicio:fin])**2))
        segmentos.append({
            'tipo': 'curva',
            'radio': RADIO_CURVA,
            'longitud': longitud_arco,
            'puntos': (x[inicio:fin], y[inicio:fin])
        })
        inicio = fin

# --- 6. Resultados ---
print("=== SEGMENTOS DETECTADOS ===")
for i, seg in enumerate(segmentos):
    if seg['tipo'] == 'recta':
        print(f"Recta {i+1}:")
        print(f"  - Longitud: {seg['longitud']:.2f} m")
        print(f"  - Ángulo: {seg['angulo']:.2f}°")
        print(f"  - Puntos: ({seg['inicio'][0]:.1f}, {seg['inicio'][1]:.1f}) a ({seg['fin'][0]:.1f}, {seg['fin'][1]:.1f})")
    else:
        print(f"Curva {i+1}:")
        print(f"  - Longitud: {seg['longitud']:.2f} m")
        print(f"  - Radio: {seg['radio']} m (fijo)")
    print("---")

# --- 7. Visualización (opcional) ---
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
for seg in segmentos:
    if seg['tipo'] == 'recta':
        plt.plot([seg['inicio'][0], seg['fin'][0]], [seg['inicio'][1], seg['fin'][1]], 'b-', linewidth=2)
    else:
        plt.plot(seg['puntos'][0], seg['puntos'][1], 'r-', linewidth=2)
plt.xlabel('X (metros)')
plt.ylabel('Y (metros)')
plt.title('Segmentos detectados: Rectas (azul) y Curvas (rojo)')
plt.grid(True)
plt.show()