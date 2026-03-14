import cv2
import numpy as np

class Deskewer:
    """
    Moduł odpowiedzialny za globalną korekcję skosu (global skew correction).
    Pomaga wyprostować zdjęcia (np. ze smartfona lub źle zeskanowane), zanim
    przekażemy je do wyciągania mikro-siatki w fazie homografii.
    """
    
    @staticmethod
    def fix_skew(image_bgr):
        """
        Wykrywa linie (Transformata Hougha) i oblicza globalny kąt obrotu.
        Zwraca wyprostowany obraz.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Opcjonalnie: możemy użyć krawędzi do znalezienia głównych osi papieru/siatki
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            # Jeśli nie znaleziono żadnych linii, zwracamy bez zmian
            return image_bgr
            
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Obliczanie kąta w stopniach (-180 do 180)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Normalizujemy kąt do przedziału [-45, 45], ponieważ siatka EKG 
            # składa się z linii ortogonalnych (różniących się o 90 stopni).
            angle = angle % 90
            if angle > 45:
                angle -= 90
                
            angles.append(angle)
            
        if not angles:
            return image_bgr
            
        # Mediana jest odporna na wartości odstające (outliers) np. krzywo napisany tekst
        median_angle = np.median(angles)
        
        # Jeśli obrót jest mniejszy niż 0.1 stopnia, szkoda tracić na jakości interpolacji
        if abs(median_angle) < 0.1:
            return image_bgr
            
        # Wykonanie rotacji afinicznej
        (h, w) = image_bgr.shape[:2]
        center = (w // 2, h // 2)
        
        # Zbudowanie macierzy rotacji M
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Obracamy; borderMode=cv2.BORDER_REPLICATE żeby nie było czarnych marginesów
        rotated = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
