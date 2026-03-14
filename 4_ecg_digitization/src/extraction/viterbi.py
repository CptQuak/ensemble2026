import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class ViterbiExtractor:
    """
    Ekstrakcja sygnału EKG z wykorzystaniem programowania dynamicznego (Viterbi Pathfinding).
    Zastępuje naiwne uśrednianie pikseli, optymalizując metrykę Shape (PCC).
    """

    @staticmethod
    def extract_signal(roi_mask, roi_gray, record_name=None, lead_name=None, debug_folder="output/debug"):
        """
        roi_mask: binarna maska sygnału (255 to sygnał)
        roi_gray: oryginalny wycinek w skali szarości (do oceny luminancji)
        """
        h, w = roi_mask.shape
        if w == 0:
            return np.zeros(0, dtype=np.float32)
        
        # 1. Definicja stanów (węzłów) w każdej kolumnie
        # Dla każdej kolumny znajdujemy klastry (bloby) aktywnych pikseli.
        # Środki tych klastrów to nasze możliwe stany w danym kroku 'x'.
        nodes = []
        for x in range(w):
            col_mask = roi_mask[:, x]
            active_y = np.where(col_mask > 0)[0]
            
            col_nodes = []
            if len(active_y) > 0:
                # Rozdzielamy ciągłe bloki pikseli (klastrowanie 1D)
                gaps = np.diff(active_y) > 2
                split_indices = np.where(gaps)[0] + 1
                clusters = np.split(active_y, split_indices)
                
                for cluster in clusters:
                    center_y = int(np.mean(cluster))
                    # Obliczamy luminancję (jasność) z oryginalnego obrazu (0 to czarny)
                    lum = int(roi_gray[center_y, x])
                    col_nodes.append({'y': center_y, 'lum': lum})
            
            # Jeśli kolumna jest pusta (przerwa w sygnale), sztucznie dodajemy 
            # węzły co kilka pikseli, aby algorytm mógł "przelecieć" przez dziurę
            if len(col_nodes) == 0:
                for y in range(0, h, 5):
                    col_nodes.append({'y': y, 'lum': 255}) # wysoka luminancja = wysoka kara
                    
            nodes.append(col_nodes)

        # 2. Inicjalizacja macierzy kosztów (Viterbi DP)
        # dp[x][i] = minimalny koszt dotarcia do i-tego węzła w kolumnie x
        dp = [np.full(len(n), np.inf) for n in nodes]
        parent = [np.zeros(len(n), dtype=int) for n in nodes]
        
        # Koszt początkowy dla pierwszej kolumny (bazuje tylko na luminancji)
        for i, node in enumerate(nodes[0]):
            dp[0][i] = node['lum'] * 0.1

        # Wagi funkcji kosztu (C = w1*dystans + w2*luminancja)
        w_dist = 1.0
        w_lum = 0.5
        
        # 3. Krok Forward: Wypełnianie macierzy kosztów
        for x in range(1, w):
            prev_nodes = nodes[x-1]
            curr_nodes = nodes[x]
            
            for i, curr_node in enumerate(curr_nodes):
                best_cost = np.inf
                best_parent = -1
                
                for j, prev_node in enumerate(prev_nodes):
                    # Odległość Y (kary za gwałtowne skoki amplitudy)
                    dist = abs(curr_node['y'] - prev_node['y'])
                    
                    # Nieliniowa kara za skok: małe zmiany są ok, duże są drastycznie karane
                    cost_dist = dist ** 2 if dist > 3 else dist
                    
                    # Luminancja (preferujemy ciemny tusz)
                    cost_lum = curr_node['lum']
                    
                    # Całkowity koszt przejścia
                    transition_cost = (w_dist * cost_dist) + (w_lum * cost_lum)
                    total_cost = dp[x-1][j] + transition_cost
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_parent = j
                        
                dp[x][i] = best_cost
                parent[x][i] = best_parent

        # 4. Krok Backward: Odtworzenie optymalnej ścieżki
        path_y = np.zeros(w, dtype=np.float32)
        
        # Zaczynamy od węzła o najmniejszym koszcie w ostatniej kolumnie
        curr_idx = np.argmin(dp[-1])
        
        for x in range(w-1, -1, -1):
            path_y[x] = nodes[x][curr_idx]['y']
            curr_idx = parent[x][curr_idx]

        # 5. Diagnostyka (zapis wizualizacji)
        if record_name and lead_name:
            path_dir = os.path.join(debug_folder, record_name)
            os.makedirs(path_dir, exist_ok=True)
            plt.figure(figsize=(10, 2))
            # Odwracamy oś Y, ponieważ obraz ma 0 na górze
            plt.plot(h - path_y, color='blue', linewidth=1) 
            plt.title(f"Viterbi Pathfinding - {lead_name}")
            plt.savefig(f"{path_dir}/step4_viterbi_{lead_name}.png")
            plt.close()

        return path_y