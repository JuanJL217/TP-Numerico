import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar los datos del CSV
df = pd.read_csv('parametrizacion_bico_de_pato_10.csv')

# 2. Función para evaluar la curva en cualquier parámetro t
def evaluar_curva(t, df):
    """Evalúa x(t) y y(t) para un t dado"""
    segmento = df[(df['t_inicio'] <= t) & (t <= df['t_fin'])].iloc[0]
    t_rel = t - segmento['t_inicio']
    x = segmento['a_x']*t_rel**3 + segmento['b_x']*t_rel**2 + segmento['c_x']*t_rel + segmento['d_x']
    y = segmento['a_y']*t_rel**3 + segmento['b_y']*t_rel**2 + segmento['c_y']*t_rel + segmento['d_y']
    return x, y

# 3. Generar puntos equiespaciados para la curva completa
t_values = np.linspace(df['t_inicio'].min(), df['t_fin'].max(), 2000)
x_vals, y_vals = zip(*[evaluar_curva(t, df) for t in t_values])

# 4. Configuración del gráfico vectorial
plt.figure(figsize=(12, 6))
plt.plot(x_vals, y_vals, 'b-', linewidth=1.5, label='Curva paramétrica')

# 5. Opcional: Añadir vectores tangentes (cada 100 puntos)
for i in range(0, len(t_values), 100):
    t = t_values[i]
    x, y = x_vals[i], y_vals[i]
    
    # Calcular derivadas (vector tangente)
    segmento = df[(df['t_inicio'] <= t) & (t <= df['t_fin'])].iloc[0]
    t_rel = t - segmento['t_inicio']
    dx_dt = 3*segmento['a_x']*t_rel**2 + 2*segmento['b_x']*t_rel + segmento['c_x']
    dy_dt = 3*segmento['a_y']*t_rel**2 + 2*segmento['b_y']*t_rel + segmento['c_y']
    
    # Normalizar el vector tangente para visualización
    norm = np.sqrt(dx_dt**2 + dy_dt**2)
    if norm > 0:
        dx_dt, dy_dt = dx_dt/norm * 5, dy_dt/norm * 5  # Longitud 5 unidades
    
    plt.arrow(x, y, dx_dt, dy_dt, head_width=0.5, head_length=0.8, fc='red', ec='red')

plt.figure(figsize=(12, 6))
plt.plot(x_vals, y_vals, 'b-', linewidth=2)
plt.title('Curva Paramétrica Completa', fontsize=14)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.show()