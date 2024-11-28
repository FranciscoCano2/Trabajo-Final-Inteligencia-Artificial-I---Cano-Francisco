import librosa
import sounddevice as sd
import numpy as np
import json
from AudioPreprocessor import AudioPreprocessor
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import speech_recognition as sr
import random

class AudioProcessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr  # Frecuencia objetivo para resampling
        self.preprocessor = AudioPreprocessor(target_sr=target_sr)

    def extract_feature_stats(self, audio, sr):
        # Convertir a Mel Spectrogram y extraer estadísticas
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=sr // 2)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)  # Convertir a escala logarítmica (dB)
        mel_mean = np.mean(mel_spectrogram_db)
        mel_std = np.std(mel_spectrogram_db)
        mel_percentiles = np.percentile(mel_spectrogram_db, [25, 50, 75])

        # Extraer MFCCs y calcular estadísticas
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1).tolist()
        mfccs_std = np.std(mfccs, axis=1).tolist()

        # Calcular Delta y Delta-Delta MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)
        delta_mfccs_mean = np.mean(delta_mfccs, axis=1).tolist()
        delta_mfccs_std = np.std(delta_mfccs, axis=1).tolist()
        delta_delta_mfccs_mean = np.mean(delta_delta_mfccs, axis=1).tolist()
        delta_delta_mfccs_std = np.std(delta_delta_mfccs, axis=1).tolist()

        # Calcular Zero-Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0]
        zcr_mean = np.mean(zero_crossing_rate)
        zcr_std = np.std(zero_crossing_rate)

        # Calcular duración del audio
        duration = librosa.get_duration(y=audio, sr=sr)

        # Calcular el espectro de potencia y estadísticas
        power_spectrum = np.abs(librosa.stft(audio)) ** 2
        power_mean = np.mean(power_spectrum)
        power_std = np.std(power_spectrum)
        power_percentiles = np.percentile(power_spectrum, [25, 50, 75])

        return {
            "mel_spectrogram": {
                "mean": mel_mean,
                "std": mel_std,
                "percentiles": mel_percentiles.tolist()
            },
            "mfcc": {
                "mean": mfccs_mean,
                "std": mfccs_std
            },
            "delta_mfcc": {
                "mean": delta_mfccs_mean,
                "std": delta_mfccs_std
            },
            "delta_delta_mfcc": {
                "mean": delta_delta_mfccs_mean,
                "std": delta_delta_mfccs_std
            },
            "zero_crossing_rate": {
                "mean": zcr_mean,
                "std": zcr_std
            },
            "power_spectrum": {
                "mean": power_mean,
                "std": power_std,
                "percentiles": power_percentiles.tolist()
            },
            "duration": duration
        }

    def process_audio_files(self, folder_path):
        """
        Procesa los archivos de audio: extrae características, las guarda como resumen,
        transcribe los audios en texto y genera un gráfico 2D basado en las transcripciones.
        """
        try:
            transcriptions = {verdura: [] for verdura in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, verdura))}
            feature_results = {verdura: [] for verdura in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, verdura))}

            recognizer = sr.Recognizer()

            all_words_by_verdura = {}

            for verdura_folder in os.listdir(folder_path):
                verdura_path = os.path.join(folder_path, verdura_folder)
                if os.path.isdir(verdura_path):
                    print(f"Procesando archivos en: {verdura_folder}...")
                    all_words_by_verdura[verdura_folder] = []
                    for audio_file in os.listdir(verdura_path):
                        audio_path = os.path.join(verdura_path, audio_file)

                        if not audio_file.lower().endswith(('.wav', '.mp3', '.ogg', '.m4a')):
                            continue

                        # Cargar y preprocesar el audio
                        audio, sr_rate = self.preprocessor.load_audio(audio_path)
                        audio = self.preprocessor.remove_silence(audio, sr_rate)
                        audio = self.preprocessor.highpass_filter(audio, sr_rate)
                        audio = self.preprocessor.lowpass_filter(audio, sr_rate)
                        audio = self.preprocessor.resample_audio(audio, sr_rate)

                        # Extraer características resumidas
                        features = self.extract_feature_stats(audio, self.target_sr)
                        feature_results[verdura_folder].append(features)

                        # Transcribir el audio a texto
                        try:
                            with sr.AudioFile(audio_path) as source:
                                audio_data = recognizer.record(source)
                            transcription = recognizer.recognize_google(audio_data, language="es-ES")
                            transcriptions[verdura_folder].append({
                                "archivo": audio_file,
                                "transcripcion": transcription
                            })
                            # Añadir palabras transcritas a la lista general por verdura
                            all_words_by_verdura[verdura_folder].extend(transcription.split())
                        except sr.UnknownValueError:
                            print(f"No se pudo transcribir el audio: {audio_file}")
                        except sr.RequestError as e:
                            print(f"Error al conectar con el servicio de reconocimiento: {e}")
                        except Exception as e:
                            print(f"Error al procesar el archivo {audio_file}: {e}")

            # Guardar estadísticas de características por verdura
            features_folder = "audio_features"
            os.makedirs(features_folder, exist_ok=True)
            for verdura, features in feature_results.items():
                output_file = os.path.join(features_folder, f"{verdura}_features_summary.json")
                with open(output_file, "w") as f:
                    json.dump(features, f, indent=4)
                print(f"Características resumidas de {verdura} guardadas en {output_file}.")

            # Guardar transcripciones por verdura
            transcriptions_folder = "transcriptions"
            os.makedirs(transcriptions_folder, exist_ok=True)
            for verdura, texts in transcriptions.items():
                output_file = os.path.join(transcriptions_folder, f"{verdura}_transcriptions.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(texts, f, indent=4, ensure_ascii=False)
                print(f"Transcripciones de {verdura} guardadas en {output_file}.")

            # Crear gráfico basado en las palabras transcritas diferenciando por verdura
            self.create_word_graph_by_category(all_words_by_verdura)

        except Exception as e:
            print(f"Error al procesar los audios: {e}")
            
    def plot_comparative_features(self, n_components=2):
        # Cargar archivos de características
        feature_files = [f for f in os.listdir("audio_features") if f.endswith("_features_summary.json")]
        all_features = []
        labels = []

        for file in feature_files:
            verdura = file.split("_")[0]
            with open(os.path.join("audio_features", file), "r") as f:
                features = json.load(f)

            # Guardar las nuevas características y etiquetas de cada archivo
            for feat in features:
                # Selecciona características relevantes
                mel_mean = feat["mel_spectrogram"]["mean"]
                mfcc_mean = feat["mfcc"]["mean"]  # Promedio de MFCC
                delta_mean = feat["delta_mfcc"]["mean"]
                delta_delta_mean = feat["delta_delta_mfcc"]["mean"]
                zcr_mean = feat["zero_crossing_rate"]["mean"]
                duration = feat["duration"]

                # Combinar todas las características en un solo vector
                combined_features = (
                    [mel_mean] + mfcc_mean + delta_mean + delta_delta_mean + [zcr_mean, duration]
                )
                all_features.append(combined_features)
                labels.append(verdura)

        # Convertir la lista de características a un arreglo de NumPy
        all_features = np.array(all_features)

        '''# Normalizar características antes de PCA
        scaler = StandardScaler()
        all_features_normalized = scaler.fit_transform(all_features)

        # Aplicar PCA para reducir dimensiones
        pca = PCA(n_components=n_components)
        transformed_features = pca.fit_transform(all_features_normalized)

        # Guardar el modelo de escalado y PCA
        os.makedirs("models", exist_ok=True)  # Crear carpeta si no existe

        with open("models/scaler_model.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print("Modelo de escalado guardado en 'models/scaler_model.pkl'.")

        with open("models/pca_model.pkl", "wb") as f:
            pickle.dump(pca, f)
        print("Modelo PCA guardado en 'models/pca_model.pkl'.")'''

        # Cargar modelos existentes
        with open("models/scaler_model.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("models/pca_model.pkl", "rb") as f:
            pca = pickle.load(f)

        all_features_normalized = scaler.fit_transform(all_features)
        
        transformed_features = pca.fit_transform(all_features_normalized)

        # Graficar la nube de puntos con PCA en 2D
        fig, ax = plt.subplots(figsize=(12, 10))

        # Asignar un color distinto para cada tipo de verdura
        unique_labels = list(set(labels))
        colors = plt.cm.get_cmap('tab10', len(unique_labels))

        for idx, label in enumerate(unique_labels):
            # Seleccionar los puntos de la verdura actual
            label_indices = [i for i, lbl in enumerate(labels) if lbl == label]
            ax.scatter(
                transformed_features[label_indices, 0],
                transformed_features[label_indices, 1],
                label=label,
                color=colors(idx),
                alpha=0.7
            )

        ax.set_title("Comparación de Audios usando PCA (2D)")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend(title="Tipo de Verdura")
        plt.show()

    def create_word_graph_by_category(self, words_by_category):
        """
        Crea un gráfico 2D con palabras transcritas, diferenciadas por verdura:
        - Eje X: cantidad de caracteres.
        - Eje Y: densidad de la palabra (suma de valores de las letras / cantidad de caracteres).
        """
        try:
            plt.figure(figsize=(12, 8))
            categories = words_by_category.keys()
            colors = {category: (random.random(), random.random(), random.random()) for category in categories}  # Colores únicos por categoría

            for category, words in words_by_category.items():
                x = []  # Cantidad de caracteres
                y = []  # Densidad de la palabra

                for word in words:
                    word = word.lower()
                    num_chars = len(word)
                    char_sum = sum((ord(char) - 96) for char in word if 'a' <= char <= 'z')  # Calcular densidad
                    density = char_sum / num_chars if num_chars > 0 else 0

                    x.append(num_chars)
                    y.append(density)

                # Agregar puntos al gráfico con un color único para la categoría
                plt.scatter(x, y, alpha=0.7, label=category, c=[colors[category]], edgecolors='black')

            plt.title("Relación entre longitud y densidad de palabras por categoría")
            plt.xlabel("Cantidad de caracteres")
            plt.ylabel("Densidad de la palabra")
            plt.legend(title="Verdura")
            plt.grid(True)
            plt.show()

        except Exception as e:
            print(f"Error al generar el gráfico: {e}")