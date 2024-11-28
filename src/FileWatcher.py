import os
import time
import shutil
from pydub import AudioSegment
import soundfile as sf
import sounddevice as sd
import numpy as np
import speech_recognition as sr

from AudioProcessor import AudioProcessor
from BatchAnalyzer import BatchAnalyzer
import json
import pickle
import cv2

class FileWatcher:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.existing_files = set(os.listdir(folder_path))

        # Crear carpetas de destino para imágenes y audios si no existen
        self.image_folder = os.path.join(folder_path, "imagen")
        self.audio_folder = os.path.join(folder_path, "audio")
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.audio_folder, exist_ok=True)

    def monitor_for_new_files(self):
        while True:
            current_files = set(os.listdir(self.folder_path))
            
            # Filtrar solo archivos (no carpetas)
            current_files = {f for f in current_files if os.path.isfile(os.path.join(self.folder_path, f))}
            
            new_files = current_files - self.existing_files

            if new_files:
                for new_file in new_files:
                    file_path = os.path.join(self.folder_path, new_file)
                    if self.is_file_fully_written(file_path):
                        self.existing_files = current_files
                        return True  # Archivo detectado y procesado, salir del bucle

            time.sleep(2)  # Espera antes de volver a chequear
        
    def process_new_file(self, folder_path):
        """
        Procesa un archivo nuevo detectado en la carpeta especificada. 
        Ignora directorios y mueve el archivo a la carpeta correspondiente según su tipo.
        """
        # Obtener la lista de archivos en la carpeta
        new_files = os.listdir(folder_path)

        for file_name in new_files:
            # Construir la ruta completa del archivo
            file_path = os.path.join(folder_path, file_name)

            # Ignorar directorios
            if os.path.isdir(file_path):
                continue

            # Carpetas de destino para imágenes y audios
            image_folder = os.path.join(folder_path, "imagen")
            audio_folder = os.path.join(folder_path, "audio")

            # Crear las carpetas de destino si no existen
            os.makedirs(image_folder, exist_ok=True)
            os.makedirs(audio_folder, exist_ok=True)

            # Clasificar y mover el archivo según su tipo
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                destination_path = os.path.join(image_folder, file_name)
                shutil.move(file_path, destination_path)
                print(f"Nuevo archivo de imagen detectado y movido a: {destination_path}")
                self.handle_image(destination_path)  # Procesar la imagen movida
                return  # Salir después de procesar un archivo válido

            elif file_name.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a')):
                destination_path = os.path.join(audio_folder, file_name)
                shutil.move(file_path, destination_path)
                print(f"Nuevo archivo de audio detectado y movido a: {destination_path}")
                self.handle_audio(destination_path)  # Procesar el audio movido
                return  # Salir después de procesar un archivo válido

            else:
                print(f"Archivo no soportado detectado: {file_name}")
                return  # Salir después de procesar un archivo no soportado
        
    def handle_image(self, file_path):
        try:
            # Mover la imagen inicial a la carpeta "imagen"
            destination_path = os.path.join(self.image_folder, os.path.basename(file_path))
            shutil.move(file_path, destination_path)
            print(f"Imagen movida a {destination_path} para procesamiento inicial.")
            
            # Analizar la imagen y determinar la verdura más cercana
            analyzer = BatchAnalyzer(os.path.dirname(destination_path), "nueva_imagen")
            new_image_metrics = analyzer.analyze_single_image(destination_path)
            
            if new_image_metrics:
                closest_vegetable = analyzer.find_closest_average(new_image_metrics)
                print(f"La verdura más cercana a la nueva imagen es: {closest_vegetable}")
                
                # Confirmar con el usuario antes de mover la imagen
                while True:
                    confirmation = input(f"¿Es correcta la clasificación '{closest_vegetable}'? (s/n): ").lower()
                    if confirmation in {'s', 'n'}:
                        break
                    print("Por favor, ingrese 's' para sí o 'n' para no.")
                
                if confirmation == 's':
                    # Definir la subcarpeta de destino según la verdura identificada
                    final_destination_folder = os.path.join(self.image_folder, closest_vegetable)
                    os.makedirs(final_destination_folder, exist_ok=True)  # Crear la carpeta si no existe
                    
                    # Mover la imagen a la subcarpeta correspondiente
                    final_destination_path = os.path.join(final_destination_folder, os.path.basename(destination_path))
                    shutil.move(destination_path, final_destination_path)
                    print(f"Imagen movida a {final_destination_path} en la carpeta de {closest_vegetable}.")
                else:
                    # Descartar la imagen
                    os.remove(destination_path)
                    print(f"Imagen descartada: {destination_path}")
            else:
                print("No se pudieron obtener métricas de la imagen para clasificarla.")
        except Exception as e:
            print(f"Error al manejar la imagen: {e}")

    def handle_audio(self, file_path):
        """
        Maneja un archivo de audio, aplicando preprocesamiento, extracción de características
        y clasificación según la elección del usuario (por características o por texto).
        """
        try:
            # Mover archivo a la carpeta temporal de audio
            destination_path = os.path.join(self.audio_folder, os.path.basename(file_path))
            shutil.move(file_path, destination_path)
            print(f"Audio movido temporalmente a {destination_path}.")

            audio_processor = AudioProcessor(target_sr=16000)
            if destination_path.endswith('.m4a'):
                # Convertir a WAV si el formato es .m4a
                wav_path = destination_path.replace('.m4a', '.wav')
                audio = AudioSegment.from_file(destination_path, format="m4a")
                audio.export(wav_path, format="wav")
                print(f"Archivo convertido a WAV: {wav_path}")

                # Eliminar el archivo .m4a original
                os.remove(destination_path)
                print(f"Archivo original eliminado: {destination_path}")
                destination_path = wav_path

            # Preguntar al usuario cómo identificar el audio
            while True:
                print("Seleccione el método para identificar el audio:")
                print("1. Por características acústicas")
                print("2. Por texto transcripto")
                option = input("Elija una opción (1 o 2): ")
                if option in {"1", "2"}:
                    break
                else:
                    print("Opción inválida. Intente nuevamente.")

            if option == "1":
                # Identificar por características acústicas
                preprocessed_audio, sample_rate = audio_processor.preprocessor.preprocess(destination_path)
                features = audio_processor.extract_feature_stats(preprocessed_audio, sample_rate)

                # Cargar modelos de PCA y escalador previamente entrenados
                with open("models/scaler_model.pkl", "rb") as f:
                    scaler_model = pickle.load(f)

                with open("models/pca_model.pkl", "rb") as f:
                    pca_model = pickle.load(f)

                # Clasificar el audio con k-NN
                prediction = self.classify_audio_with_pca(
                    audio_features=features,
                    pca_model=pca_model,
                    scaler_model=scaler_model,
                    k=3,
                    max_neighbors=5
                )
                print(f"El audio clasificado pertenece a: {prediction}")

            elif option == "2":
                # Identificar por texto transcripto
                recognizer = sr.Recognizer()
                try:
                    with sr.AudioFile(destination_path) as source:
                        audio_data = recognizer.record(source)
                    transcription = recognizer.recognize_google(audio_data, language="es-ES")
                    print(f"Transcripción del audio: {transcription}")

                    # Calcular características textuales
                    words = transcription.split()
                    results = []
                    for word in words:
                        length = len(word)
                        density = sum((ord(char) - 96) for char in word.lower() if 'a' <= char <= 'z') / length
                        results.append({"word": word, "length": length, "density": density})

                    # Clasificar el texto transcripto con k-NN basado en palabras almacenadas
                    prediction = self.classify_text_knn(results, k=3)
                    print(f"El audio clasificado según el texto pertenece a: {prediction}")
                except sr.UnknownValueError:
                    print("No se pudo transcribir el audio.")
                    return
                except sr.RequestError as e:
                    print(f"Error al conectar con el servicio de reconocimiento: {e}")
                    return

            # Confirmar con el usuario antes de mover el archivo
            while True:
                confirmation = input(f"¿Es correcta la clasificación '{prediction}'? (s/n): ").lower()
                if confirmation in {'s', 'n'}:
                    break
                print("Por favor, ingrese 's' para sí o 'n' para no.")

            if confirmation == 's':
                # Mover el archivo a la carpeta correspondiente según la predicción
                verdura_folder = os.path.join(self.audio_folder, prediction)
                os.makedirs(verdura_folder, exist_ok=True)  # Crear la carpeta si no existe
                final_path = os.path.join(verdura_folder, os.path.basename(destination_path))
                shutil.move(destination_path, final_path)
                print(f"Audio movido a la carpeta correspondiente: {final_path}")

                # Mostrar imágenes de la carpeta correspondiente
                verdura_image_folder = os.path.join(self.image_folder, prediction)
                if os.path.exists(verdura_image_folder):
                    self.display_images_in_folder(verdura_image_folder)
                else:
                    print(f"No se encontraron imágenes en la carpeta: {verdura_image_folder}")
            else:
                # Descartar el archivo
                os.remove(destination_path)
                print(f"Audio descartado: {destination_path}")

        except Exception as e:
            print(f"Error al manejar el audio: {e}")


    def handle_audio_with_processing(self, folder_path):
        # Crear instancia de AudioProcessor
        audio_processor = AudioProcessor(target_sr=16000)

        # Crear carpeta temporal para procesar el audio
        temp_path = "temp_audio_processing"
        os.makedirs(temp_path, exist_ok=True)

        try:
            # Limpiar carpeta temporal al inicio del procesamiento
            for old_file in os.listdir(temp_path):
                os.remove(os.path.join(temp_path, old_file))

            # Verificar archivos en la carpeta proporcionada
            files = os.listdir(folder_path)
            if not files:
                print("No se encontraron archivos en la carpeta.")
                return

            # Obtener el archivo más reciente en la carpeta
            file_path = os.path.join(folder_path, max(files, key=lambda f: os.path.getctime(os.path.join(folder_path, f))))
            print(f"Procesando archivo más reciente: {file_path}")  # Depuración para verificar el archivo seleccionado

            # Verificar si el archivo está completamente escrito
            if not self.is_file_fully_written(file_path):
                print(f"El archivo '{file_path}' aún no está completamente escrito. Reintentando más tarde.")
                return

            # Verificar si el archivo existe antes de moverlo
            if not os.path.exists(file_path):
                print(f"El archivo '{file_path}' no existe o ya fue movido.")
                return

            # Generar un nombre único para la carpeta temporal si existe conflicto
            temp_file_path = os.path.join(temp_path, os.path.basename(file_path))
            if os.path.exists(temp_file_path):
                base, ext = os.path.splitext(temp_file_path)
                temp_file_path = f"{base}_temp{ext}"

            print(f"Intentando mover el archivo a la carpeta temporal: {temp_file_path}")

            try:
                shutil.move(file_path, temp_file_path)
                print(f"Archivo movido temporalmente a: {temp_file_path}")
            except Exception as move_error:
                print(f"Error al mover el archivo: {move_error}")
                return

            # Solicitar tipo de verdura al usuario
            verdura = input("Ingrese el tipo de verdura para el audio (berenjena, camote, papa, zanahoria): ").strip().lower()
            if verdura not in ["berenjena", "camote", "papa", "zanahoria"]:
                print("Tipo de verdura no reconocido. Archivo descartado.")
                os.remove(temp_file_path)
                return

            # Convertir a WAV si el archivo es .m4a
            destination_path = temp_file_path  # Inicialmente igual a temp_file_path
            if temp_file_path.endswith('.m4a'):
                wav_path = temp_file_path.replace('.m4a', '.wav')
                audio = AudioSegment.from_file(temp_file_path, format="m4a")
                audio.export(wav_path, format="wav")
                destination_path = wav_path  # Actualizar destination_path al nuevo archivo WAV
                print(f"Archivo convertido a WAV: {wav_path}")

            # Preprocesar y cargar el audio
            preprocessed_audio, sample_rate = audio_processor.preprocessor.preprocess(destination_path)

            # Extraer características acústicas
            features = audio_processor.extract_feature_stats(preprocessed_audio, sample_rate)

            # Cargar modelos de PCA y escalador previamente entrenados
            with open("models/scaler_model.pkl", "rb") as f:
                scaler_model = pickle.load(f)

            with open("models/pca_model.pkl", "rb") as f:
                pca_model = pickle.load(f)

            # Clasificar el audio con k-NN
            prediction = self.classify_audio_with_pca(
                audio_features=features,
                pca_model=pca_model,
                scaler_model=scaler_model,
                k=3,
                max_neighbors=5
            )
            print(f"Predicción del modelo k-NN: {prediction}")

            # Comparar predicción con la entrada del usuario
            if prediction == verdura:
                print("La predicción del modelo coincide con la entrada del usuario.")
            else:
                print("La predicción del modelo NO coincide con la entrada del usuario.")

            # Ciclo para decidir si guardar o descartar el audio procesado
            while True:
                keep = input(f"¿Desea guardar este audio en la carpeta de la verdura indicada ({verdura})? (s/n): ").strip().lower()
                if keep == "n":
                    os.remove(destination_path)
                    print("Audio descartado.")
                    break
                elif keep == "s":
                    # Mover el archivo a la carpeta de la verdura
                    final_destination_folder = os.path.join(self.audio_folder, verdura)
                    os.makedirs(final_destination_folder, exist_ok=True)
                    final_destination_path = os.path.join(final_destination_folder, os.path.basename(destination_path))
                    shutil.move(destination_path, final_destination_path)
                    print(f"Audio guardado en {final_destination_path}.")
                    break
                else:
                    print("Opción no válida, intente nuevamente.")

        except Exception as e:
            print(f"Error al procesar el audio: {e}")

        finally:
            # Limpiar la carpeta temporal
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
            print("Carpeta temporal eliminada.")



    def is_file_fully_written(self, file_path, wait_time=2):
        """Verifica si un archivo está completamente escrito comprobando si su tamaño deja de cambiar."""
        initial_size = -1
        while True:
            try:
                current_size = os.path.getsize(file_path)
                if current_size == initial_size:
                    return True  # El archivo ya no está cambiando de tamaño
                initial_size = current_size
                time.sleep(wait_time)
            except (FileNotFoundError, PermissionError):
                return False
            except Exception as e:
                print(f"Error al verificar el archivo: {e}")
                return False

    def classify_audio_with_pca(self, audio_features, pca_model, scaler_model, k=3, max_neighbors=5):
        """
        :param audio_features: Diccionario con las características del audio a clasificar.
        :param pca_model: Modelo PCA previamente entrenado con los datos de entrenamiento.
        :param scaler_model: Escalador previamente ajustado con los datos de entrenamiento.
        :param k: Número inicial de vecinos más cercanos a considerar.
        :param max_neighbors: Máximo número de vecinos a considerar si no hay consenso.
        :return: Predicción de la clase del audio.
        """
        try:
            # Cargar las características y etiquetas de entrenamiento
            feature_files = [f for f in os.listdir("audio_features") if f.endswith("_features_summary.json")]
            all_features = []
            labels = []

            for file in feature_files:
                verdura = file.split("_")[0]
                with open(os.path.join("audio_features", file), "r") as f:
                    features = json.load(f)
                for feat in features:
                    mel_mean = feat["mel_spectrogram"]["mean"]
                    mfcc_mean = feat["mfcc"]["mean"]  # Promedio de MFCC
                    delta_mean = feat["delta_mfcc"]["mean"]
                    delta_delta_mean = feat["delta_delta_mfcc"]["mean"]
                    zcr_mean = feat["zero_crossing_rate"]["mean"]
                    duration = feat["duration"]

                    combined_features = (
                        [mel_mean] + mfcc_mean + delta_mean + delta_delta_mean + [zcr_mean, duration]
                    )
                    all_features.append(combined_features)
                    labels.append(verdura)

            # Convertir las características a un arreglo de NumPy
            all_features = np.array(all_features)

            # Normalizar las características de entrenamiento usando el escalador proporcionado
            all_features_normalized = scaler_model.transform(all_features)

            # Aplicar PCA a los datos de entrenamiento
            pca_features = pca_model.transform(all_features_normalized)

            # Parametrizar las características del nuevo audio siguiendo el mismo procedimiento
            mel_mean = audio_features["mel_spectrogram"]["mean"]
            mfcc_mean = audio_features["mfcc"]["mean"]
            delta_mean = audio_features["delta_mfcc"]["mean"]
            delta_delta_mean = audio_features["delta_delta_mfcc"]["mean"]
            zcr_mean = audio_features["zero_crossing_rate"]["mean"]
            duration = audio_features["duration"]

            combined_features = (
                [mel_mean] + mfcc_mean + delta_mean + delta_delta_mean + [zcr_mean, duration]
            )

            # Normalizar y transformar las características del nuevo audio
            new_audio_normalized = scaler_model.transform([combined_features])
            new_audio_pca = pca_model.transform(new_audio_normalized)

            # Calcular distancias euclidianas entre el nuevo audio y los datos de entrenamiento transformados
            distances = []
            for idx, feature in enumerate(pca_features):
                distance = np.sqrt(np.sum((feature - new_audio_pca[0]) ** 2))
                distances.append((distance, labels[idx]))

            # Ordenar por distancia (ascendente)
            distances.sort(key=lambda x: x[0])

            # Evaluar consenso
            nearest_neighbors = [distances[i][1] for i in range(k)]
            prediction = max(set(nearest_neighbors), key=nearest_neighbors.count)
            return prediction

        except Exception as e:
            print(f"Error al clasificar el audio: {e}")
            return None


    def display_images_in_folder(self, folder_path):
        """
        Muestra todas las imágenes en la carpeta especificada. 
        Las ventanas se cierran al presionar la barra espaciadora.
        """
        try:
            # Listar todas las imágenes en la carpeta
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if not image_files:
                print(f"No se encontraron imágenes en la carpeta: {folder_path}")
                return

            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"No se pudo leer la imagen: {image_path}")
                    continue

                # Redimensionar la imagen
                resized_image = self.resize_image_to_fit_screen(image)

                cv2.imshow(f"Imagen: {image_file}", resized_image)

            print("Presiona la barra espaciadora para cerrar las imágenes.")
            while True:
                key = cv2.waitKey(0)
                if key == 32:  # Código ASCII para la barra espaciadora
                    break

            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error al mostrar las imágenes: {e}")

    def resize_image_to_fit_screen(self, image, max_width=800, max_height=600):
        """
        Redimensiona la imagen para que se ajuste a la pantalla o un tamaño fijo.
        Mantiene la relación de aspecto.
        """
        try:
            height, width = image.shape[:2]
            scaling_factor = min(max_width / width, max_height / height)

            # Calcular el nuevo tamaño
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)

            # Redimensionar la imagen
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized_image

        except Exception as e:
            print(f"Error al redimensionar la imagen: {e}")
            return image  # En caso de error, devolver la imagen original
        
    def classify_text_knn(self, text_features, k=3):
        """
        Clasifica texto basado en características textuales con k-NN, utilizando palabras
        transcritas previamente almacenadas en archivos JSON.
        """
        try:
            # Cargar palabras transcriptas desde archivos en la carpeta 'transcriptions'
            transcription_folder = "transcriptions"
            word_data = []

            for transcription_file in os.listdir(transcription_folder):
                file_path = os.path.join(transcription_folder, transcription_file)
                if transcription_file.endswith("_transcriptions.json"):
                    with open(file_path, "r") as f:
                        transcriptions = json.load(f)

                    # Extraer información de cada palabra transcripta
                    for entry in transcriptions:
                        word = entry["transcripcion"]
                        length = len(word)
                        density = sum((ord(char) - 96) for char in word.lower() if 'a' <= char <= 'z') / length
                        category = transcription_file.replace("_transcriptions.json", "")  # Categoría según el archivo
                        word_data.append({"word": word, "length": length, "density": density, "category": category})

            # Calcular distancias entre las características de entrada y las palabras almacenadas
            distances = []
            for entry in word_data:
                for feature in text_features:
                    dist = np.sqrt(
                        (entry["length"] - feature["length"]) ** 2 +
                        (entry["density"] - feature["density"]) ** 2
                    )
                    distances.append((dist, entry["category"]))

            # Ordenar por distancia y tomar los k más cercanos
            distances.sort(key=lambda x: x[0])
            nearest_neighbors = distances[:k]

            # Contar las categorías más frecuentes
            categories = [neighbor[1] for neighbor in nearest_neighbors]
            prediction = max(set(categories), key=categories.count)
            return prediction

        except Exception as e:
            print(f"Error al clasificar el texto: {e}")
            return None