import librosa
import numpy as np
from scipy.signal import butter, filtfilt

class AudioPreprocessor:
    def __init__(self, target_sr=16000, silence_top_db=20, highpass_cutoff=1000, lowpass_cutoff=3000, filter_order=5):
        # Configuración de parámetros de procesamiento
        self.target_sr = target_sr                # Frecuencia objetivo para resampling
        self.silence_top_db = silence_top_db      # Umbral para detección de silencio
        self.highpass_cutoff = highpass_cutoff    # Frecuencia de corte del filtro pasa-alto
        self.lowpass_cutoff = lowpass_cutoff      # Frecuencia de corte del filtro pasa-bajo
        self.filter_order = filter_order          # Orden de los filtros

    def load_audio(self, file_path):
        # Cargar audio desde el archivo sin resampling
        audio, sr = librosa.load(file_path, sr=None)
        return audio, sr

    def remove_silence(self, audio, sr):
        # Quitar tiempos muertos del audio según el umbral `silence_top_db`
        intervals = librosa.effects.split(audio, top_db=self.silence_top_db)
        processed_audio = np.concatenate([audio[start:end] for start, end in intervals])
        return processed_audio

    def highpass_filter(self, audio, sr):
        # Filtro pasa-alto para reducir bajas frecuencias
        nyquist = 0.5 * sr
        normal_cutoff = self.highpass_cutoff / nyquist
        b, a = butter(self.filter_order, normal_cutoff, btype='high', analog=False)
        filtered_audio = filtfilt(b, a, audio)
        return filtered_audio

    def lowpass_filter(self, audio, sr):
        # Filtro pasa-bajo para reducir altas frecuencias
        nyquist = 0.5 * sr
        normal_cutoff = self.lowpass_cutoff / nyquist
        b, a = butter(self.filter_order, normal_cutoff, btype='low', analog=False)
        filtered_audio = filtfilt(b, a, audio)
        return filtered_audio

    def resample_audio(self, audio, original_sr):
        # Resampling del audio a la frecuencia objetivo `target_sr`
        return librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)

    def preprocess(self, file_path):
        # Cargar y procesar el audio
        audio, sr = self.load_audio(file_path)
        
        # Aplicar cada paso de procesamiento
        audio = self.remove_silence(audio, sr)
        audio = self.highpass_filter(audio, sr)
        audio = self.lowpass_filter(audio, sr)
        audio = self.resample_audio(audio, sr)
        
        return audio, self.target_sr
