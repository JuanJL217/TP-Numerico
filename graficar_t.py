import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar los datos de parametrización
df = pd.read_csv('parametrizacion_bico_de_pato_10.csv')

# 2. Función para evaluar la curva en cualquier t
def evaluar_curva(t):
    seg = df[(df['t_inicio'] <= t) & (t <= df['t_fin'])].iloc[0]
    t_rel = t - seg['t_inicio']
    x = seg['a_x']*t_rel**3 + seg['b_x']*t_rel**2 + seg['c_x']*t_rel + seg['d_x']
    y = seg['a_y']*t_rel**3 + seg['b_y']*t_rel**2 + seg['c_y']*t_rel + seg['d_y']
    return x, y

# 3. Generar puntos para graficar
t_values = np.linspace(df['t_inicio'].min(), df['t_fin'].max(), 1000)
x_values = []
y_values = []

for t in t_values:
    x, y = evaluar_curva(t)
    x_values.append(x)
    y_values.append(y)

# 4. Configurar el gráfico
plt.figure(figsize=(12, 6))

# Gráfico de X vs t
plt.subplot(2, 1, 1)
plt.plot(t_values, x_values, 'b-', linewidth=2)
plt.title('Coordenada X en función del parámetro t')
plt.xlabel('Parámetro t')
plt.ylabel('X(t)')
plt.grid(True)

# Gráfico de Y vs t
plt.subplot(2, 1, 2)
plt.plot(t_values, y_values, 'r-', linewidth=2)
plt.title('Coordenada Y en función del parámetro t')
plt.xlabel('Parámetro t')
plt.ylabel('Y(t)')
plt.grid(True)

plt.tight_layout()
plt.show()

# 5. Gráfico de la curva en el plano XY (opcional)
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, 'g-', linewidth=2)
plt.title('Curva paramétrica en el plano XY')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')  # Misma escala en ambos ejes
plt.show()