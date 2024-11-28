import cv2
import numpy as np

class ContourDetector:
    def __init__(self, image_path, block_size=6):
        self.image_path = image_path
        self.block_size = block_size
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"No se pudo cargar la imagen en {self.image_path}")

        # Definimos los colores base para la reconstrucción (BGR)
        self.base_colors = {
            "rojo": (0, 0, 255),
            "verde": (0, 255, 0),
            "azul": (255, 0, 0),
            "amarillo": (0, 255, 255),
            "cian": (255, 255, 0),
            "magenta": (255, 0, 255),
            "blanco": (255, 255, 255),
            "negro": (0, 0, 0),
        }

    def homogenize_image(self):
        result_image = self.image.copy()
        for y in range(0, result_image.shape[0], self.block_size):
            for x in range(0, result_image.shape[1], self.block_size):
                block = result_image[y:y + self.block_size, x:x + self.block_size]
                mean_color = block.mean(axis=(0, 1))
                if np.mean(mean_color) > 220:
                    nearest_color = (255, 255, 255)
                elif np.mean(mean_color) < 30:
                    nearest_color = (0, 0, 0)
                else:
                    nearest_color = self.find_nearest_base_color(mean_color)
                result_image[y:y + self.block_size, x:x + self.block_size] = nearest_color
        return result_image

    def find_nearest_base_color(self, color):
        min_dist = float("inf")
        nearest_color = None
        for base_name, base_color in self.base_colors.items():
            dist = np.linalg.norm(np.array(base_color) - np.array(color))
            if dist < min_dist:
                min_dist = dist
                nearest_color = base_color
        return nearest_color

    def binarize_image(self, processed_image, threshold=128):
        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        inverted_image = cv2.bitwise_not(binary_image)
        return inverted_image

    def remove_small_components(self, binarized_image, min_area=100):
        contours, _ = cv2.findContours(binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_image = np.zeros_like(binarized_image)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.drawContours(filtered_image, [contour], -1, 255, thickness=cv2.FILLED)
        return filtered_image

    def calculate_mean_colors(self, contour):
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        mean_colors = cv2.mean(self.image, mask=mask)
        return mean_colors[2], mean_colors[1], mean_colors[0]

    def find_and_draw_largest_contour(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Encuentra el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calcular métricas básicas
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Calcular el aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h != 0 else 0

        # Calcular los momentos de Hu
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()  # Devuelve una lista de 7 valores

        # Suponiendo que tienes métodos para calcular los valores de color
        red, green, blue = self.calculate_mean_colors(largest_contour)

        # Guardar todas las métricas en el diccionario de resultados
        return {
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "hu_moments": hu_moments.tolist(),  # Convertir a lista para fácil manejo
            "red": red,
            "green": green,
            "blue": blue
        }
