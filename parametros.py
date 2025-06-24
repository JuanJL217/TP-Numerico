import numpy as np
import cv2
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

# --- 1. Cargar imagen y detectar la curva ---
img = cv2.imread('BicoDePatoSquema.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Ajustar estos valores según el color de tu curva
lower_color = np.array([0, 0, 200])
upper_color = np.array([180, 50, 255])
mask = cv2.inRange(hsv, lower_color, upper_color)

# Encontrar contornos y extraer puntos
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
curve_points = contours[0].squeeze()

# Escalar puntos a coordenadas reales
x = curve_points[:, 0] * (100 / img.shape[1])  # X ∈ [0, 100]
y = (img.shape[0] - curve_points[:, 1]) * (80 / img.shape[0])  # Y ∈ [0, 80]

# Suavizar datos
x_smooth = savgol_filter(x, window_length=15, polyorder=2)
y_smooth = savgol_filter(y, window_length=15, polyorder=2)

# --- 2. Parametrización con splines cúbicos ---
t = np.linspace(0, 1, len(x_smooth))
cs_x = CubicSpline(t, x_smooth)
cs_y = CubicSpline(t, y_smooth)

# --- 3. Preparar datos para CSV ---
# Transponer coeficientes para obtener shape (n_segmentos, 4)
coef_x = cs_x.c.T
coef_y = cs_y.c.T

# Crear lista de diccionarios con los datos de cada segmento
segment_data = []
step = 1  # Saltar cada 2 segmentos (ajusta este valor según necesites)

for i in range(0, len(cs_x.x) - 1, step):  # Nota: el -1 debe permanecer
    t_start = cs_x.x[i]
    t_end = cs_x.x[i + step] if (i + step) < len(cs_x.x) else cs_x.x[-1]  # Evita desborde
    
    segment_data.append({
        'segmento': (i // step) + 1,  # Nuevo número de segmento
        't_inicio': t_start,
        't_fin': t_end,
        'a_x': coef_x[i, 0],
        'b_x': coef_x[i, 1],
        'c_x': coef_x[i, 2],
        'd_x': coef_x[i, 3],
        'a_y': coef_y[i, 0],
        'b_y': coef_y[i, 1],
        'c_y': coef_y[i, 2],
        'd_y': coef_y[i, 3]
    })

# Crear DataFrame y guardar como CSV
df = pd.DataFrame(segment_data)
csv_filename = 'parametrizacion_bico_de_pato_10.csv'
df.to_csv(csv_filename, index=False, float_format='%.8f')

# --- 4. Mostrar confirmación ---
print(f"\nParametrización guardada en '{csv_filename}'")
print(f"Número total de segmentos: {len(df)}")
print("\nPrimeras filas del archivo CSV:")
print(df.head())

# Mostrar ejemplo de cómo usar los coeficientes
print("\nEjemplo de uso para el primer segmento:")
print(f"Para t ∈ [{df.iloc[0]['t_inicio']:.4f}, {df.iloc[0]['t_fin']:.4f}]:")
print("x(t) = {:.8f}·(t-{:.4f})³ + {:.8f}·(t-{:.4f})² + {:.8f}·(t-{:.4f}) + {:.8f}".format(
    df.iloc[0]['a_x'], df.iloc[0]['t_inicio'],
    df.iloc[0]['b_x'], df.iloc[0]['t_inicio'],
    df.iloc[0]['c_x'], df.iloc[0]['t_inicio'],
    df.iloc[0]['d_x']))
print("y(t) = {:.8f}·(t-{:.4f})³ + {:.8f}·(t-{:.4f})² + {:.8f}·(t-{:.4f}) + {:.8f}".format(
    df.iloc[0]['a_y'], df.iloc[0]['t_inicio'],
    df.iloc[0]['b_y'], df.iloc[0]['t_inicio'],
    df.iloc[0]['c_y'], df.iloc[0]['t_inicio'],
    df.iloc[0]['d_y']))