import json
import os

from ImageTrainer import ImageTrainer
from BatchAnalyzer import BatchAnalyzer
from MetricPlotter import MetricPlotter
from FileWatcher import FileWatcher
from AudioProcessor import AudioProcessor

# Función para calcular y guardar promedios en base a los resultados
def calculate_and_store_averages(results, verdura):
    num_images = len(results)
    if num_images > 0:
        # Inicialización de acumuladores para promedios
        total_area = total_perimeter = total_circularity = total_red = 0
        total_green = total_blue = total_aspect_ratio = 0
        total_hu_moments = [0] * 7

        # Acumular valores de cada métrica de todas las imágenes
        for metrics in results:
            total_area += metrics.get("area", 0)
            total_perimeter += metrics.get("perimeter", 0)
            total_circularity += metrics.get("circularity", 0)
            total_red += metrics.get("red", 0)
            total_green += metrics.get("green", 0)
            total_blue += metrics.get("blue", 0)
            total_aspect_ratio += metrics.get("aspect_ratio", 0)
            for i in range(7):
                total_hu_moments[i] += metrics.get("hu_moments", [0] * 7)[i]

        # Calcular promedios
        average_results = {
            "area": total_area / num_images,
            "perimeter": total_perimeter / num_images,
            "circularity": total_circularity / num_images,
            "red": total_red / num_images,
            "green": total_green / num_images,
            "blue": total_blue / num_images,
            "aspect_ratio": total_aspect_ratio / num_images,
            "hu_moments": [hu / num_images for hu in total_hu_moments]
        }

        # Guardar promedios en un archivo específico para la verdura
        promedio_file_path = os.path.join("promedios", f"promedios_resultados_{verdura}.json")
        with open(promedio_file_path, "w") as f:
            json.dump(average_results, f, indent=4)
        print(f"Promedios guardados en {promedio_file_path}")
        
        return average_results  # Retorna los resultados promedio para graficarlos
    else:
        print("No hay imágenes para calcular promedios.")
        return None

# Función principal del programa
def main():

    while True:
        
        print("\nOpciones:")
        print("1. Entrenamiento de audio")
        print("2. Entrenamiento de imágenes")
        print("3. Analizar imágenes y guardar resultados")
        print("4. Leer datos desde un archivo de resultados")
        print("5. Graficar promedios de todas las verduras")
        print("6. Graficar nube de puntos de todas las verduras")
        print("7. Agregar archivo nuevo (monitorizar carpeta)")
        print("8. Analizar audios y extraer características")
        print("9. Salir")
        option = input("Seleccione una opción (1-9): ").strip()

        if option == "1":
            audio_folder_to_watch = "audios"
            file_watcher = FileWatcher(audio_folder_to_watch)

            keep_monitoring = True
            print(f"Monitorizando la carpeta '{audio_folder_to_watch}' en espera de nuevos archivos de audio...")

            while keep_monitoring:
                if file_watcher.monitor_for_new_files():

                    file_watcher.handle_audio_with_processing(audio_folder_to_watch)
                    
                    # Preguntar si el usuario desea seguir monitoreando
                    user_choice = input("¿Desea seguir monitoreando la carpeta? (s/n): ").strip().lower()
                    keep_monitoring = (user_choice == 's')
        
        elif option == "2":
            # Llamar al método de entrenamiento de imágenes
            trainer = ImageTrainer("verduras")  # Instancia de la clase que maneja entrenamiento
            trainer.train_image()
        
        elif option == "3" or option == "4":
            verdura = input("Ingrese el tipo de verdura a analizar (berenjena, camote, papa, zanahoria): ").strip().lower()
            if verdura not in ["berenjena", "camote", "papa", "zanahoria"]:
                print("Tipo de verdura no reconocido. Intente nuevamente.")
                return

            folder_path = os.path.join("verduras", verdura)  # Ruta dinámica según verdura
            analyzer = BatchAnalyzer(folder_path, verdura)

            if option == "3":
                results = analyzer.analyze_images()  # Procesa imágenes y guarda resultados
                print("Análisis completado y datos guardados en archivo.")
            elif option == "4":
                results = analyzer.load_results_from_file()  # Carga datos desde archivo de la verdura
                if not results:
                    print("No se encontraron datos en el archivo.")
                    return

            # Preguntar al usuario si desea calcular y guardar promedios
            calcular_promedios = input("¿Desea calcular y guardar los promedios? (s/n): ").strip().lower()
            if calcular_promedios == "s":
                average_results = calculate_and_store_averages(results, verdura)

            # Preguntar al usuario si desea mostrar gráficos
            mostrar_graficos = input("¿Desea mostrar gráficos de los datos promediados? (s/n): ").strip().lower()
            if mostrar_graficos == "s":
                if average_results:
                    MetricPlotter.plot_metrics_average(average_results)
                else:
                    print("No hay valores promedio para graficar.")
            else:
                # Mostrar gráficos con los datos sin promediar
                MetricPlotter.plot_metrics(results)

        elif option == "5":
            # Cargar y graficar los promedios de todas las verduras
            MetricPlotter.plot_all_averages()

        elif option == "6":
            # Cargar y graficar la nube de puntos con los datos de las verdura
            MetricPlotter.plot_3d_scatter()

        elif option == "7":
            # Monitorear una carpeta para detectar archivos nuevos
            folder_to_watch = "Archivos Nuevos"
            watcher = FileWatcher(folder_to_watch)
            print(f"Monitorizando la carpeta '{folder_to_watch}' en espera de nuevos archivos...")

            keep_monitoring = True
            while keep_monitoring:
                if watcher.monitor_for_new_files():
                    watcher.process_new_file(folder_to_watch)
                    
                    # Preguntar si el usuario desea seguir monitoreando
                    user_choice = input("¿Desea seguir monitoreando la carpeta? (s/n): ").strip().lower()
                    keep_monitoring = (user_choice == 's')
        
        elif option == "8":
            # Procesar audios y extraer características
            audio_folder_path = "audios/audio"  # Ruta donde se encuentran los audios
            processor = AudioProcessor()
            processor.process_audio_files(audio_folder_path)
            processor.plot_comparative_features()   

        elif option == "9":
            print("Saliendo del programa...")
            break
    

if __name__ == "__main__":
    main()
