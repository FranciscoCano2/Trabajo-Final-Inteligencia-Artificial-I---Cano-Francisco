import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json

class MetricPlotter:
    @staticmethod
    def plot_metrics(results):
        areas = [result["area"] for result in results]
        perimeters = [result["perimeter"] for result in results]
        circularities = [result["circularity"] for result in results]
        reds = [result["red"] for result in results]
        greens = [result["green"] for result in results]
        blues = [result["blue"] for result in results]
        aspect_ratios = [result["aspect_ratio"] for result in results]
        hu_moments = [result["hu_moments"] for result in results]  # Lista de listas de momentos de Hu
        indices = [f"{i+1}" for i in range(len(results))]  # Usamos índices en lugar de nombres de archivo

        # Lista de todos los gráficos para dividir en ventanas de 3 en 3
        plots = [
            (indices, areas, "Área", 'blue'),
            (indices, perimeters, "Perímetro", 'green'),
            (indices, circularities, "Circularidad", 'purple'),
            (indices, reds, "Rojo", 'red'),
            (indices, greens, "Verde", 'lime'),
            (indices, blues, "Azul", 'cyan'),
            (indices, aspect_ratios, "Aspect Ratio", 'orange')
        ]
        
        # Agregamos cada Hu Momento a la lista de gráficos
        for i in range(7):
            moment_values = [hu[i] for hu in hu_moments]
            plots.append((indices, moment_values, f"Hu Momento {i+1}", None))

        # Mostrar 3 gráficos por ventana
        for i in range(0, len(plots), 3):
            plt.figure(figsize=(12, 8))
            for j in range(3):
                if i + j < len(plots):  # Verifica que no se exceda el índice
                    x, y, title, color = plots[i + j]
                    plt.subplot(1, 3, j + 1)
                    plt.bar(x, y, color=color)
                    plt.title(title)
            plt.tight_layout()
            plt.show()

    def plot_metrics_average(results):
        # Extraer los datos del diccionario de promedios
        areas = results["area"] 
        perimeters = results["perimeter"] 
        circularities = results["circularity"] 
        reds = results["red"] 
        greens = results["green"] 
        blues = results["blue"] 
        aspect_ratios = results["aspect_ratio"] 
        hu_moments = results["hu_moments"]  # Lista de listas de momentos de Hu 

        # Ventana uno: área, perímetro, circularidad, aspect_ratio
        plots_window_one = [
            (["1"], [areas], "Área", 'blue'), 
            (["1"], [perimeters], "Perímetro", 'green'), 
            (["1"], [circularities], "Circularidad", 'purple'), 
            (["1"], [aspect_ratios], "Aspect Ratio", 'orange')
        ]
        
        # Ventana dos: rojo, verde, azul
        plots_window_two = [
            (["1"], [reds], "Rojo", 'red'), 
            (["1"], [greens], "Verde", 'lime'), 
            (["1"], [blues], "Azul", 'cyan')
        ]

        # Ventana tres: momentos de Hu
        plots_window_three = []
        for i in range(7): 
            moment_values = [hu_moments[i]]  # Solo un valor por Hu Momento 
            plots_window_three.append((["1"], moment_values, f"Hu Momento {i + 1}", None))

        # Mostrar gráficos en ventanas
        # Ventana uno
        plt.figure(figsize=(12, 8)) 
        for j in range(len(plots_window_one)):
            x, y, title, color = plots_window_one[j]
            plt.subplot(1, 4, j + 1)  # Cambié a 1 fila y 4 columnas
            plt.bar(x, y, color=color)
            plt.title(title)
        plt.tight_layout()
        plt.show()

        # Ventana dos
        plt.figure(figsize=(12, 8))
        for j in range(len(plots_window_two)):
            x, y, title, color = plots_window_two[j]
            plt.subplot(1, 3, j + 1)  # 1 fila y 3 columnas
            plt.bar(x, y, color=color)
            plt.title(title)
        plt.tight_layout()
        plt.show()

        # Ventana tres
        plt.figure(figsize=(12, 8))
        for j in range(len(plots_window_three)):
            x, y, title, color = plots_window_three[j]
            plt.subplot(3, 3, j + 1)  # 3 filas y 3 columnas (para 7 momentos)
            plt.bar(x, y, color=color if color else 'gray')  # Color gris si no se especifica
            plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_all_averages():
        vegetables = ["berenjena", "camote", "papa", "zanahoria"]
        all_averages = []

        # Cargar los archivos de promedios para cada verdura
        for vegetable in vegetables:
            promedio_file_path = os.path.join("promedios", f"promedios_resultados_{vegetable}.json")
            try:
                with open(promedio_file_path, "r") as f:
                    data = json.load(f)
                    all_averages.append((vegetable, data))
            except FileNotFoundError:
                print(f"Archivo de promedios para {vegetable} no encontrado.")

        if not all_averages:
            print("No se encontraron datos de promedios para graficar.")
            return

        # Extraer datos para graficar
        characteristics = ["area", "perimeter", "circularity", "red", "green", "blue", "aspect_ratio"]
        hu_moments = [f"Hu Momento {i+1}" for i in range(7)]

        # Graficar cada característica por separado
        for characteristic in characteristics:
            plt.figure(figsize=(10, 6))
            for vegetable, data in all_averages:
                plt.bar(vegetable, data[characteristic], label=vegetable)
            plt.title(f"Comparación de {characteristic.capitalize()} entre verduras")
            plt.ylabel(characteristic.capitalize())
            plt.xlabel("Verduras")
            plt.tight_layout()
            plt.show()

        # Graficar momentos de Hu
        for i in range(7):
            plt.figure(figsize=(10, 6))
            for vegetable, data in all_averages:
                plt.bar(vegetable, data["hu_moments"][i], label=vegetable)
            plt.title(f"Comparación de Hu Momento {i+1} entre verduras")
            plt.ylabel(f"Hu Momento {i+1}")
            plt.xlabel("Verduras")
            plt.tight_layout()
            plt.show()


    @staticmethod
    def plot_3d_scatter():
        # Lista de las verduras y sus colores correspondientes
        vegetables = ["berenjena", "camote", "papa", "zanahoria"]
        colors = {
            "berenjena": "purple",
            "camote": "orange",
            "papa": "brown",
            "zanahoria": "green"
        }

        # Diccionarios para almacenar las características de cada verdura
        green_values = {veg: [] for veg in vegetables}
        blue_values = {veg: [] for veg in vegetables}
        red_values = {veg: [] for veg in vegetables}

        # Cargar datos de cada verdura desde los archivos detallados
        for vegetable in vegetables:
            resultados_file_path = os.path.join("resultados", f"resultados_analisis_{vegetable}.json")
            try:
                with open(resultados_file_path, "r") as f:
                    data_list = json.load(f)
                    if not isinstance(data_list, list):
                        raise ValueError(f"Se esperaba una lista en {resultados_file_path}, pero se obtuvo {type(data_list)}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error al leer o procesar {resultados_file_path}: {e}")
                continue

            # Extraer las características de cada imagen
            for data in data_list:
                if isinstance(data, dict):
                    green_values[vegetable].append(data.get("green", 0))
                    blue_values[vegetable].append(data.get("blue", 0))  # Extraer el valor de 'blue'
                    red_values[vegetable].append(data.get("red", 0))
                else:
                    print(f"Advertencia: Se encontró un elemento no diccionario en {resultados_file_path}")

        # Crear la gráfica de nube de puntos en 3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Agregar los puntos al gráfico con colores y etiquetas correspondientes
        for vegetable in vegetables:
            ax.scatter(
                green_values[vegetable],
                blue_values[vegetable],
                red_values[vegetable],
                c=colors[vegetable],
                label=vegetable,
                marker='o'
            )

        # Calcular y graficar los centroides de cada grupo
        for vegetable in vegetables:
            mean_green = sum(green_values[vegetable]) / len(green_values[vegetable])
            mean_blue = sum(blue_values[vegetable]) / len(blue_values[vegetable])
            mean_red = sum(red_values[vegetable]) / len(red_values[vegetable])
            
            # Graficar el centroide con un marcador grande y bordes negros
            ax.scatter(
                mean_green, mean_blue, mean_red,
                c=colors[vegetable],
                edgecolor="black",  # Bordes negros para distinguir
                s=100,  # Tamaño del marcador más grande
                label=f"Centroide {vegetable}",
                marker='X'
            )

        # Etiquetas de los ejes
        ax.set_xlabel("Green")
        ax.set_ylabel("Blue")
        ax.set_zlabel("Red")

        # Título del gráfico
        ax.set_title("Nube de Puntos 3D: Green, Blue, Red")

        # Agregar la leyenda para los colores de cada verdura y sus centroides
        ax.legend(title="Verdura", loc="best")

        # Mostrar el gráfico
        plt.show()

    def plot_histogram(self, data, title="Histograma", xlabel="Valor", ylabel="Frecuencia", bins=10):
        """
        Grafica un histograma a partir de los datos proporcionados.
        """
        plt.figure()
        plt.hist(data, bins=bins, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_line_chart(self, x_data, y_data, title="Gráfico de Línea", xlabel="Índice", ylabel="Valor"):
        """
        Grafica un gráfico de líneas.
        """
        plt.figure()
        plt.plot(x_data, y_data, marker='o', linestyle='-')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_bar_chart(self, categories, values, title="Gráfico de Barras", xlabel="Categorías", ylabel="Valores"):
        """
        Grafica un gráfico de barras.
        """
        plt.figure()
        plt.bar(categories, values, color='skyblue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_scatter(self, x_data, y_data, title="Gráfico de Dispersión", xlabel="Eje X", ylabel="Eje Y"):
        """
        Grafica un gráfico de dispersión.
        """
        plt.figure()
        plt.scatter(x_data, y_data, color='green')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    