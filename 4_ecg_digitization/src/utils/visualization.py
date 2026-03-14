import cv2
import numpy as np
import os

class ECGVisualizer:
    """Moduł diagnostyczny do weryfikacji wierności morfologicznej (Shape)."""
    
    @staticmethod
    def save_debug_image(image, path):
        """Zapisuje obraz diagnostyczny do folderu output/ w formacie .png."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)

    @staticmethod
    def create_overlay(image_bgr, signal_px, y_offset=0, color=(255, 0, 255), thickness=3):
        """
        Nakłada fioletową linię sygnału na obraz.
        image_bgr: Oryginalne zdjęcie EKG.
        signal_px: Wektor pozycji Y w pikselach.
        y_offset: Przesunięcie, jeśli analizujemy konkretne ROI.
        """
        overlay = image_bgr.copy()
        width = len(signal_px)
        x_coords = np.arange(width)
        
        valid_idx = ~np.isnan(signal_px)
        if not np.any(valid_idx):
            return overlay
            
        x_coords_valid = x_coords[valid_idx]
        signal_px_valid = signal_px[valid_idx]
        
        # Przygotowanie punktów (x, y) uwzględniając offset ROI
        points = np.column_stack((x_coords_valid, signal_px_valid + y_offset)).astype(np.int32)
        
        # Rysowanie polilinii - wizualna weryfikacja PCC
        cv2.polylines(overlay, [points], isClosed=False, color=color, thickness=thickness)
        return overlay