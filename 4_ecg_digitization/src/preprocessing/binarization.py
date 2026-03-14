import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class ECGPreprocessor:
    """Moduł przygotowania obrazu z wizualizacją kroków pośrednich."""
    
    @staticmethod
    def save_debug_image(image, path):
        """Zapisuje obraz w postaci .png dla podanego etapu."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)

    @staticmethod
    def process(image_bgr, record_name, debug_folder="output/debug"):
        path_prefix = os.path.join(debug_folder, record_name)
        
        # KROK 1: Kanał Zielony - wybieramy kanał zielony (standard dla EEG)
        green = image_bgr[:, :, 1]
        ECGPreprocessor.save_debug_image(green, f"{path_prefix}/step1_green_channel.png")

        # KROK 2: Binarizacja Otsu
        _, binary = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ECGPreprocessor.save_debug_image(binary, f"{path_prefix}/step2_binary_otsu.png")

        # KROK 3: Usuwanie szumu (Operacja otwarcia) (Salt and Pepper)
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        ECGPreprocessor.save_debug_image(cleaned, f"{path_prefix}/step3_cleaned_mask.png")
        return cleaned
