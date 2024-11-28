import os
import json
import math
from ContourDetector import ContourDetector
from ImageSelector import ImageSelector

class BatchAnalyzer:
    def __init__(self, folder_path, vegetable):
        self.folder_path = folder_path
        self.vegetable = vegetable
        self.file_path = os.path.join("resultados", f"resultados_analisis_{self.vegetable}.json")

    def analyze_images(self):
        results = []
        selector = ImageSelector(self.folder_path)

        for image_info in selector.list_images():
            image_path = image_info["path"]
            image_index = image_info["index"]

            detector = ContourDetector(image_path)
            homogenized_image = detector.homogenize_image()
            binarized_image = detector.binarize_image(homogenized_image)
            filtered_image = detector.remove_small_components(binarized_image)

            metrics = detector.find_and_draw_largest_contour(filtered_image)

            if metrics:
                metrics["index"] = image_index
                results.append(metrics)

        # Guardar resultados individuales en el archivo JSON
        with open(self.file_path, "w") as f:
            json.dump(results, f, indent=4)

        return results

    def load_results_from_file(self):
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
                # Verificar si el dato cargado es una lista o un diccionario
                if isinstance(data, dict):
                    return data.get("results", [])
                elif isinstance(data, list):
                    return data  # Retornar directamente la lista si es el caso
                else:
                    print("Formato de datos inesperado en el archivo.")
                    return []
        except FileNotFoundError:
            print("Archivo de resultados no encontrado.")
            return []

    def analyze_single_image(self, image_path):
        """
        Analiza una sola imagen y devuelve las métricas calculadas.
        """
        detector = ContourDetector(image_path)
        homogenized_image = detector.homogenize_image()
        binarized_image = detector.binarize_image(homogenized_image)
        filtered_image = detector.remove_small_components(binarized_image)

        metrics = detector.find_and_draw_largest_contour(filtered_image)
        return metrics  # Retorna el diccionario con las métricas de la imagen analizada


    def find_closest_average(self, new_image_metrics):
        """
        Calcula la distancia euclidiana entre los parámetros de la imagen nueva y los promedios de las verduras,
        y devuelve el nombre de la verdura con la menor distancia.
        """
        # Cargar los valores promedio de las verduras
        average_metrics = self.load_average_metrics()

        # Seleccionar los parámetros que usaremos para calcular la distancia
        parameters = ["green", "blue", "red"]  # Cambia a los nombres reales de tus parámetros

        closest_vegetable = None
        min_distance = float('inf')

        for vegetable, averages in average_metrics.items():
            # Calcular la distancia euclidiana solo para los parámetros seleccionados
            distance = math.sqrt(sum(
                (new_image_metrics[param] - averages[param]) ** 2 for param in parameters
            ))

            print(f"Distancia a {vegetable}: {distance}")

            if distance < min_distance:
                min_distance = distance
                closest_vegetable = vegetable

        print(f"La verdura más cercana en promedio es: {closest_vegetable}")
        return closest_vegetable
    
    def load_average_metrics(self):
        """
        Carga los archivos de promedios de todas las verduras y devuelve un diccionario.
        """
        average_metrics = {}
        for file_name in os.listdir("promedios"):
            if file_name.endswith(".json"):
                verdura = file_name.split("_")[-1].replace(".json", "")
                with open(os.path.join("promedios", file_name), "r") as f:
                    average_metrics[verdura] = json.load(f)
        return average_metrics
    