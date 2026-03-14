import cv2
import numpy as np
import os

class GridDetector:
    """
    Faza 1: Bezwzględna Normalizacja Przestrzenna.
    Moduł odpowiedzialny za detekcję siatki milimetrowej EKG i wyznaczanie węzłów.
    Wymagane dla metryki Amplitude (SNR) i Time Calibration.
    """
    
    @staticmethod
    def extract_grid_mask(image_bgr, record_name=None, debug_folder="output/debug"):
        """
        Wyodrębnia maskę binarną siatki EKG (najczęściej różowej/czerwonej).
        Siatka jest dobrze widoczna, gdy odejmiemy kanał czerwony od niebieskiego/zielonego,
        lub pracując w przestrzeni HSV/LAB.
        """
        # 1. Konwersja do przestrzeni HSV, gdzie łatwiej odizolować kolor siatki (róż/czerwień)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        
        # Zakresy dla koloru różowego/czerwonego (typowe dla papieru EKG)
        # Uwaga: Czerwień w HSV jest na brzegach (0-10 i 170-180)
        lower_red1 = np.array([0, 30, 100])
        upper_red1 = np.array([20, 255, 255])
        lower_red2 = np.array([160, 30, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        grid_color_mask = mask1 | mask2
        
        # Alternatywna metoda (robustna na wyblakły tusz): 
        # Różnica między kanałem zielonym (gdzie sygnał EKG i siatka są ciemne) 
        # a czerwonym (gdzie siatka jest jasna, a sygnał ciemny).
        b, g, r = cv2.split(image_bgr)
        # Wzmocnienie różnicy: czerwony jest jasny na siatce, a zielony ciemny.
        # Różnica (R - G) będzie duża dla pikseli siatki.
        diff_rg = cv2.subtract(r, g)
        _, grid_diff_mask = cv2.threshold(diff_rg, 15, 255, cv2.THRESH_BINARY)
        
        # Łączymy obie metody dla maksymalnej skuteczności (logic OR)
        combined_grid_mask = cv2.bitwise_or(grid_color_mask, grid_diff_mask)
        
        # Oczyszczanie maski siatki (usuwanie szumu, np. tekstów)
        kernel = np.ones((2, 2), np.uint8)
        cleaned_grid = cv2.morphologyEx(combined_grid_mask, cv2.MORPH_OPEN, kernel)
        
        if record_name:
            path_prefix = os.path.join(debug_folder, record_name)
            os.makedirs(path_prefix, exist_ok=True)
            cv2.imwrite(f"{path_prefix}/step1a_grid_color_mask.png", grid_color_mask)
            cv2.imwrite(f"{path_prefix}/step1b_grid_diff_mask.png", grid_diff_mask)
            cv2.imwrite(f"{path_prefix}/step1c_grid_combined.png", cleaned_grid)
            
        return cleaned_grid

    @staticmethod
    def detect_grid_lines(grid_mask, record_name=None, debug_folder="output/debug"):
        """
        Wykrywa główne (major - 5mm) i pomocnicze (minor - 1mm) linie siatki 
        używając morfologii matematycznej i transformaty Hougha.
        """
        # 1. Izolacja linii pionowych i poziomych za pomocą długich kerneli morfologicznych
        # Zakładamy, że linie są stosunkowo proste na małych odcinkach
        min_line_length = 50 
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_length))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, 1))
        
        vertical_lines = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, vertical_kernel)
        horizontal_lines = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Odtworzenie pełnych linii (pogrubienie po erozji w MORPH_OPEN)
        vertical_lines = cv2.dilate(vertical_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
        horizontal_lines = cv2.dilate(horizontal_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        
        if record_name:
            path_prefix = os.path.join(debug_folder, record_name)
            cv2.imwrite(f"{path_prefix}/step1d_vertical_lines.png", vertical_lines)
            cv2.imwrite(f"{path_prefix}/step1e_horizontal_lines.png", horizontal_lines)
            
        return vertical_lines, horizontal_lines

    @staticmethod
    def find_intersections(vertical_lines, horizontal_lines, record_name=None, debug_folder="output/debug"):
        """
        Znajduje węzły siatki (przecięcia linii pionowych i poziomych).
        """
        # Przecięcie to logiczne AND między maską linii pionowych i poziomych
        intersections_mask = cv2.bitwise_and(vertical_lines, horizontal_lines)
        
        # Znajdowanie środków klastrów (węzłów)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersections_mask, connectivity=8)
        
        # Filtrujemy szum (zbyt małe lub zbyt duże klastry)
        valid_centroids = []
        for i in range(1, num_labels): # Pomijamy tło (i=0)
            area = stats[i, cv2.CC_STAT_AREA]
            if 4 <= area <= 100: # Węzeł powinien być małym punktem (ok. 2x2 do 10x10 px)
                valid_centroids.append(centroids[i])
                
        valid_centroids = np.array(valid_centroids)
        
        if record_name and len(valid_centroids) > 0:
            path_prefix = os.path.join(debug_folder, record_name)
            
            # Tworzymy kolorowy obraz z zaznaczonymi węzłami
            debug_img = cv2.cvtColor(intersections_mask, cv2.COLOR_GRAY2BGR)
            for cx, cy in valid_centroids:
                cv2.circle(debug_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                
            cv2.imwrite(f"{path_prefix}/step1f_intersections.png", debug_img)
            
        return valid_centroids
