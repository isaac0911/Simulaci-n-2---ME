import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit

# Parámetros de la simulación (cambiar en función de lo que se desee)
N_values = [10] # Lista de valores de N (tamaño de la red)
t_values = [1] # Lista de valores de t (cantidad inicial de cuantos por espacio)
iterations = [10000] # Número total de iteraciones de la simulación
update_interval = [1000] # Intervalo en el que se actualiza la visualización


# Función para regresión exponencial
def exponential(x, alpha, beta):
    return alpha * np.exp(-beta * x)  # Función exponencial: α * exp(-β * x)

# Función de simulación
def boltzmann_simulation(N, t, iterations, update_interval):
    # Inicializar la red
    grid = np.full((N, N), t)  # Crear una red N x N con t cuantos en cada espacio
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))  # Crear un gráfico con dos subplots

    # Función para actualizar y graficar
    def update(frame):
        for _ in range(update_interval):
            # Elige dos posiciones al azar
            x1, y1 = np.random.randint(0, N, 2)  # Primera posición aleatoria en la red
            x2, y2 = np.random.randint(0, N, 2)  # Segunda posición aleatoria en la red

            # Transferir un cuanto de energía
            if grid[x1, y1] > 0:  # Verifica si la primera posición tiene al menos un cuanto
                grid[x1, y1] -= 1  # Resta un cuanto de energía de la primera posición
                grid[x2, y2] += 1  # Añade un cuanto de energía a la segunda posición

        ax[0].clear()  # Limpia el histograma anterior
        ax[1].clear()  # Limpia la visualización de la red anterior

        # Histograma
        flat_grid = grid.flatten()  # Aplana la red en un array 1D
        ax[0].hist(flat_grid, bins=range(np.max(flat_grid) + 2), edgecolor='black')  # Grafica el histograma
        ax[0].set_xlabel('Número de cuantos')  # Etiqueta del eje x
        ax[0].set_ylabel('Cantidad de espacios')  # Etiqueta del eje y
        ax[0].set_title(f'Histograma (N={N}, t={t}) - Iteración {(frame+1) * update_interval}')  # Título del histograma

        # Mostrar la red si N <= 20
        if N <= 20:
            ax[1].imshow(grid, cmap='hot', interpolation='nearest')  # Muestra la red de espacios
            ax[1].set_title('Red de espacios')  # Título de la red
            for i in range(N):
                for j in range(N):
                    ax[1].text(j, i, str(grid[i, j]), ha='center', va='center', color='white')  # Muestra el número de cuantos en cada celda

        plt.tight_layout()  # Ajusta el espaciado del gráfico

    ani = FuncAnimation(fig, update, frames=range(iterations // update_interval), repeat=False)  # Anima la simulación
    plt.show()  # Muestra la animación

    # Histograma final y ajuste exponencial
    flat_grid = grid.flatten()  # Aplana la red para el histograma final
    counts, bins = np.histogram(flat_grid, bins=range(np.max(flat_grid) + 2))  # Crea el histograma final

    # Ajuste exponencial
    bins = bins[:-1]  # Elimina el último bin, que no tiene datos asociados
    popt, _ = curve_fit(exponential, bins, counts, p0=(counts[0], 0.1))  # Ajusta la función exponencial a los datos del histograma

    plt.figure(figsize=(8, 5))  # Crea una nueva figura para el ajuste exponencial
    plt.scatter(bins, counts, label='Datos')  # Muestra los datos del histograma como puntos
    plt.plot(bins, exponential(bins, *popt), 'r-', label=f'Ajuste: α={popt[0]:.2f}, β={popt[1]:.2f}')  # Grafica la curva ajustada
    plt.xlabel('Número de cuantos')  # Etiqueta del eje x
    plt.ylabel('Cantidad de espacios')  # Etiqueta del eje y
    plt.title(f'Ajuste exponencial de la distribución de cuantos (N={N}, t={t}) - {iter} Iteraciones')  # Título del gráfico
    plt.legend()  # Muestra la leyenda
    plt.show()  # Muestra el gráfico final

    return grid, popt  # Retorna la red final y los parámetros ajustados

# Ejecutar la simulación para diferentes valores de N y t
for N, iter, updt_inter in zip(N_values, iterations, update_interval):
    for t in t_values:
        print(f'Simulación para N={N}, t={t}')  # Imprime el estado actual de la simulación
        grid, popt = boltzmann_simulation(N, t, iter, updt_inter)  # Ejecuta la simulación
        print(f'Ajuste exponencial: α={popt[0]:.2f}, β={popt[1]:.2f}')  # Muestra los resultados del ajuste exponencial


        