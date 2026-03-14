import cv2
import numpy as np
import os

class ECGPreprocessor:
    """Moduł wieloetapowego przygotowania obrazu."""
    
    @staticmethod
    def save_debug_image(image, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)

    @staticmethod
    def get_segmentation_mask(image_bgr, record_name, debug_folder="output/debug"):
        """
        Maska dla ECGSegmenter.
        Używa adaptacyjnej binaryzacji. W przeciwieństwie do globalnego Otsu,
        adaptacyjna binaryzacja (Gaussian) świetnie radzi sobie z nierównomiernym 
        oświetleniem (zagięcia, cienie na papierze), zachowując grubą siatkę,
        impulsy kalibracyjne i separatory, co pozwala poprawnie wyliczyć wariancję.
        """
        path_prefix = os.path.join(debug_folder, record_name)
        
        green = image_bgr[:, :, 1]
        
        # Adaptacyjna binaryzacja: okno 51x51, stała C=15 (odporność na nierówne oświetlenie)
        binary = cv2.adaptiveThreshold(
            green, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            51, 15
        )
        
        # Lekkie czyszczenie szumu Salt & Pepper
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        ECGPreprocessor.save_debug_image(cleaned, f"{path_prefix}/step2_segmentation_mask.png")
        return cleaned

    @staticmethod
    def get_extraction_mask(image_bgr, record_name, debug_folder="output/debug"):
        """
        Maska i mapa luminancji dla ViterbiExtractor.
        Wykorzystuje separację kanałów (Max Channel Trick), aby całkowicie
        wymazać różową/czerwoną siatkę milimetrową i zostawić tylko czarny tusz.
        """
        path_prefix = os.path.join(debug_folder, record_name)
        
        # Rozbicie na kanały
        b, g, r = cv2.split(image_bgr)
        
        # Trik: bierzemy najjaśniejszą składową. Różowa siatka staje się biała, czarny tusz zostaje czarny.
        gray_no_grid = cv2.max(cv2.max(b, g), r)
        ECGPreprocessor.save_debug_image(gray_no_grid, f"{path_prefix}/step2a_gray_no_grid.png")
        
        # Binarizacja na obrazie już pozbawionym siatki
        _, binary = cv2.threshold(gray_no_grid, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Zabezpieczenie przed przerwaniem cienkich linii EKG
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.dilate(cleaned, np.ones((2, 2), np.uint8), iterations=1)
        
        ECGPreprocessor.save_debug_image(cleaned, f"{path_prefix}/step2b_extraction_mask.png")
        
        # Zwracamy maskę ORAZ obraz bez siatki w skali szarości (jako mapę kosztu luminancji)
        return cleaned, gray_no_grid