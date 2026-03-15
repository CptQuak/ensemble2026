import cv2
import numpy as np

class HomographyWarp:
    """
    Wykonuje Lokalną Transformację Homograficzną (DLT) na wycinkach obrazu.
    Pozwala to na mapowanie zniekształconej siatki (np. odkształconej przez 
    obiektyw lub pognieciony papier) na idealną cyfrową siatkę.
    """
    
    @staticmethod
    def undistort_quad(image, pts_src, pts_dst, dsize):
        """
        Przekształca 4 punkty z obrazu wejściowego (pts_src) 
        na 4 idealne punkty docelowe (pts_dst).
        
        Zwraca: wycięty i wyprostowany kwadrat o rozmiarze `dsize`.
        """
        # Obliczenie macierzy homografii H = R^{3x3}
        H, _ = cv2.findHomography(pts_src, pts_dst, method=0)
        
        # Aplikacja rzutowania DLT (Direct Linear Transform)
        warped = cv2.warpPerspective(image, H, dsize, flags=cv2.INTER_LINEAR)
        return warped
    
    @staticmethod
    def undistort_image_grid(image_bgr, nodes_matrix, cell_size_mm=5, target_px_per_mm=20):
        rows, cols, _ = nodes_matrix.shape
        
        # ZABEZPIECZENIE: Jeśli siatka ma mniej niż 10 linii w dowolnym kierunku, nie da się bezpiecznie jej użyć
        if rows < 10 or cols < 10:
            print("  [Warning] Zbyt mało węzłów siatki do homografii. Zwracam oryginał.")
            return image_bgr

        target_cell_px = int(cell_size_mm * target_px_per_mm)

        dst_width = int((cols - 1) * target_cell_px)
        dst_height = int((rows - 1) * target_cell_px)

        # Jeśli znaleziona siatka jest za mała w stosunku do obrazu, ignorujemy ją, 
        # bo moglibyśmy drastycznie zmniejszyć rozdzielczość sygnału
        if dst_width < image_bgr.shape[1] * 0.5 or dst_height < image_bgr.shape[0] * 0.5:
            print("  [Warning] Wykryta siatka jest zbyt mała. Zwracam oryginał.")
            return image_bgr
        
        # Dodatkowe sprawdzenie przed alokacją pamięci
        if dst_width <= 0 or dst_height <= 0:
            return image_bgr

        undistorted_image = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
        
        for r in range(rows - 1):
            for c in range(cols - 1):
                # 4 punkty źródłowe (lewy-górny, prawy-górny, lewy-dolny, prawy-dolny)
                pts_src = np.array([
                    nodes_matrix[r, c],
                    nodes_matrix[r, c+1],
                    nodes_matrix[r+1, c],
                    nodes_matrix[r+1, c+1]
                ], dtype=np.float32)
                
                # 4 punkty docelowe (idealny kwadrat)
                pts_dst = np.array([
                    [0, 0],
                    [target_cell_px, 0],
                    [0, target_cell_px],
                    [target_cell_px, target_cell_px]
                ], dtype=np.float32)
                
                # Zabezpieczenie przed węzłami (0,0) które oznaczają brak węzła
                if np.any(pts_src == 0):
                    continue
                    
                # Wyprostowanie kwadratu
                warped_cell = HomographyWarp.undistort_quad(
                    image_bgr, pts_src, pts_dst, (target_cell_px, target_cell_px)
                )
                
                # Wstawienie wyprostowanego kawałka w odpowiednie miejsce na obrazie docelowym
                y_start = r * target_cell_px
                x_start = c * target_cell_px
                undistorted_image[y_start:y_start+target_cell_px, x_start:x_start+target_cell_px] = warped_cell
                
        return undistorted_image
