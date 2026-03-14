import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class GridBuilder:
    """
    Faza 1: Rekonstrukcja topologiczna siatki (Gridder).
    Bierze zaszumione, niekompletne linie i węzły, a następnie buduje
    matematycznie spójną macierz węzłów (rows, cols, 2) do transformacji DLT.
    """
    
    @staticmethod
    def _find_line_centers(line_mask, axis, min_distance=20):
        """
        Projektuje maskę linii na oś 1D i znajduje lokalne maksima (środki linii).
        axis=0 dla linii pionowych (rzutowanie na oś X)
        axis=1 dla linii poziomych (rzutowanie na oś Y)
        """
        from scipy.signal import find_peaks
        
        # Rzutowanie (suma pikseli)
        projection = np.sum(line_mask, axis=axis)
        
        # Wygładzanie
        smoothed = np.convolve(projection, np.ones(5)/5, mode='same')
        
        # Szukanie pików
        peaks, properties = find_peaks(smoothed, distance=min_distance, height=np.mean(smoothed)*0.5)
        
        return peaks, smoothed

    @staticmethod
    def build_grid_matrix(vertical_lines_mask, horizontal_lines_mask, record_name=None, debug_folder="output/debug"):
        """
        Tworzy pełną, dwuwymiarową macierz węzłów siatki, interpolując brakujące punkty.
        Zwraca: nodes_matrix o kształcie (num_horizontal_lines, num_vertical_lines, 2)
        Zawiera (x, y) dla każdego węzła.
        """
        # KROK 1: Estymacja idealnych linii siatki (przybliżenie liniowe dla całego dokumentu)
        # Znajdujemy bazowe pozycje linii pionowych (x) i poziomych (y)
        # Przyjmujemy min_distance=15 pikseli, co odpowiada ok. 1mm w 400 DPI (ok 15-20px)
        # W zależności od rozdzielczości skanu, ten parametr musi być dynamiczny, ale na razie stały.
        x_peaks, x_proj = GridBuilder._find_line_centers(vertical_lines_mask, axis=0, min_distance=15)
        y_peaks, y_proj = GridBuilder._find_line_centers(horizontal_lines_mask, axis=1, min_distance=15)
        
        if record_name:
            path_prefix = os.path.join(debug_folder, record_name)
            os.makedirs(path_prefix, exist_ok=True)
            
            # Zapisz rzutowania diagnostyczne
            plt.figure(figsize=(12, 4))
            plt.subplot(121)
            plt.plot(x_proj)
            plt.scatter(x_peaks, x_proj[x_peaks], color='red')
            plt.title("Vertical Lines Projection (X-axis)")
            
            plt.subplot(122)
            plt.plot(y_proj)
            plt.scatter(y_peaks, y_proj[y_peaks], color='red')
            plt.title("Horizontal Lines Projection (Y-axis)")
            
            plt.savefig(f"{path_prefix}/step1g_grid_projections.png")
            plt.close()

        # KROK 2: Budowa macierzy węzłów (Grid Matrix)
        # Na tym etapie tworzymy idealną, sztywną siatkę na bazie znalezionych pików.
        # W pełnej implementacji "Griddera" z artykułu, tutaj odbywałoby się śledzenie krzywizny
        # każdej pojedynczej linii (np. polifit 2 stopnia dla pogniecionego papieru).
        # Implementujemy wersję solidną (sztywna interpolacja), która zapobiega dziurom.
        
        num_rows = len(y_peaks)
        num_cols = len(x_peaks)
        
        nodes_matrix = np.zeros((num_rows, num_cols, 2), dtype=np.float32)
        
        # Wypełnianie macierzy idealnymi przecięciami
        # W zaawansowanej wersji tutaj szukalibyśmy lokalnego przesunięcia (tzw. "snapping")
        # węzła do najbliższego "białego" piksela w intersections_mask.
        for r, y in enumerate(y_peaks):
            for c, x in enumerate(x_peaks):
                nodes_matrix[r, c] = [x, y]
                
        # KROK 3: "Snapping" (Dociąganie) i Interpolacja
        # Dociągamy idealne punkty do fizycznych przecięć z tolerancją kilku pikseli
        intersections_mask = cv2.bitwise_and(vertical_lines_mask, horizontal_lines_mask)
        
        snapped_matrix = np.copy(nodes_matrix)
        search_radius = 5
        
        for r in range(num_rows):
            for c in range(num_cols):
                cx, cy = int(nodes_matrix[r, c, 0]), int(nodes_matrix[r, c, 1])
                
                # Wycinamy małe okienko wokół idealnego węzła
                y_min, y_max = max(0, cy - search_radius), min(intersections_mask.shape[0], cy + search_radius + 1)
                x_min, x_max = max(0, cx - search_radius), min(intersections_mask.shape[1], cx + search_radius + 1)
                
                window = intersections_mask[y_min:y_max, x_min:x_max]
                
                # Jeśli w okienku jest jakiś węzeł (biały piksel), przesuwamy nasz punkt
                if np.any(window):
                    # Znajdź środek masy w okienku
                    M = cv2.moments(window)
                    if M["m00"] != 0:
                        dx = int(M["m10"] / M["m00"]) - search_radius
                        dy = int(M["m01"] / M["m00"]) - search_radius
                        snapped_matrix[r, c] = [cx + dx, cy + dy]

        if record_name:
            # Wizualizacja odtworzonej i dociągniętej siatki
            debug_img = np.zeros((*intersections_mask.shape, 3), dtype=np.uint8)
            
            # Rysujemy linie poziome
            for r in range(num_rows):
                pts = snapped_matrix[r, :, :].astype(np.int32)
                cv2.polylines(debug_img, [pts], False, (0, 255, 0), 1)
                
            # Rysujemy linie pionowe
            for c in range(num_cols):
                pts = snapped_matrix[:, c, :].astype(np.int32)
                cv2.polylines(debug_img, [pts], False, (0, 255, 0), 1)
                
            # Rysujemy węzły
            for r in range(num_rows):
                for c in range(num_cols):
                    pt = tuple(snapped_matrix[r, c].astype(int))
                    cv2.circle(debug_img, pt, 2, (0, 0, 255), -1)
                    
            cv2.imwrite(f"{path_prefix}/step1h_reconstructed_grid.png", debug_img)

        return snapped_matrix
