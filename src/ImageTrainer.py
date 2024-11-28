import os
import shutil
import time
from BatchAnalyzer import BatchAnalyzer
from FileWatcher import FileWatcher

class ImageTrainer:
    def __init__(self, image_folder="verduras"):
        self.image_folder = image_folder
        self.file_watcher = FileWatcher(image_folder)  # Instancia de FileWatcher
        self.existing_files = set(os.listdir(image_folder))  # Archivos actuales en la carpeta

    
    def monitor_for_new_files(self):
        """
        Monitoriza la carpeta de imágenes y detecta nuevos archivos.
        """
        while True:
            current_files = set(os.listdir(self.image_folder))
            new_files = current_files - self.existing_files

            if new_files:
                for new_file in new_files:
                    file_path = os.path.join(self.image_folder, new_file)
                    if self.file_watcher.is_file_fully_written(file_path):
                        self.existing_files = current_files  # Actualiza el estado de archivos existentes
                        return file_path  # Devuelve la ruta del archivo detectado

            time.sleep(2)  # Espera antes de volver a chequear


    def train_image(self):
        """
        Monitoriza la carpeta de imágenes para nuevos archivos, analiza las imágenes y
        permite al usuario confirmar o corregir la clasificación.
        """
        print("Monitorizando carpeta 'verduras' para nuevas imágenes...")

        while True:
            # Detectar un nuevo archivo en la carpeta
            new_image_path = self.monitor_for_new_files()

            print(f"Imagen detectada: {new_image_path}")
            analyzer = BatchAnalyzer(self.image_folder, "nueva_imagen")
            new_image_metrics = analyzer.analyze_single_image(new_image_path)

            if not new_image_metrics:
                print("No se pudieron obtener métricas de la imagen. Imagen descartada.")
                continue

            # Determinar la verdura más cercana
            closest_vegetable = analyzer.find_closest_average(new_image_metrics)
            print(f"Clasificación inicial: {closest_vegetable}")

            # Confirmar o corregir la clasificación
            is_correct = input(f"¿La clasificación es correcta? ({closest_vegetable}) (s/n): ").strip().lower()

            if is_correct == "n":
                correct_vegetable = input("Indique la verdura correcta (berenjena, camote, papa, zanahoria): ").strip().lower()
                if correct_vegetable not in ["berenjena", "camote", "papa", "zanahoria"]:
                    print("Clasificación no válida. Imagen descartada.")
                    os.remove(new_image_path)  # Eliminar imagen si no se especifica correctamente
                    continue
            else:
                correct_vegetable = closest_vegetable

            # Preguntar si se desea almacenar la imagen
            store_image = input(f"¿Desea almacenar la imagen en la carpeta de {correct_vegetable}? (s/n): ").strip().lower()
            if store_image == "s":
                destination_folder = os.path.join(self.image_folder, correct_vegetable)
                os.makedirs(destination_folder, exist_ok=True)
                destination_path = os.path.join(destination_folder, os.path.basename(new_image_path))
                shutil.move(new_image_path, destination_path)
                print(f"Imagen almacenada en: {destination_path}")
            else:
                os.remove(new_image_path)  # Eliminar la imagen si no se desea almacenar
                print("Imagen descartada.")

            # Preguntar si continuar monitorizando
            continue_monitoring = input("¿Desea seguir monitorizando para nuevas imágenes? (s/n): ").strip().lower()
            if continue_monitoring == "n":
                print("Finalizando monitoreo de imágenes.")
                break
