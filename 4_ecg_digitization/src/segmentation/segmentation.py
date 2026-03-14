import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class ECGSegmenter:
    """Moduł do segmentacji obrazu EKG na poszczególne rzędy i odprowadzenia."""

    @staticmethod
    def save_debug_image(image, path):
        """Zapisuje wycięty segment obrazu jako .png."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)

    @staticmethod
    def find_horizontal_rows(binary_mask, record_name=None, debug_folder="output/debug"):
        """
        Wykrywa rzędy odprowadzeń analizując wariancję wierszy.
        Wiersze z sygnałem EKG mają znacznie wyższą wariancję niż puste tło lub ramki.
        """
        from scipy.signal import find_peaks
        import matplotlib.pyplot as plt

        # 1. Obliczanie wariancji dla każdego wiersza (lepiej wykrywa 'ruch' sygnału niż suma)
        row_stats = np.var(binary_mask, axis=1)

        # 2. Wygładzanie sygnału statystyk (Gaussian-like smoothing)
        # Pomaga usunąć piki wywołane pojedynczymi szumami lub kropkami siatki
        window_size = 50
        smoothed_stats = np.convolve(row_stats, np.ones(window_size)/window_size, mode='same')

        # 3. Dynamiczne szukanie pików (środków rzędów)
        # min_distance: rzędy nie mogą być bliżej siebie niż 12.5% wysokości obrazu (ok. 300px dla 2500px)
        min_distance = binary_mask.shape[0] // 8
        peaks, _ = find_peaks(smoothed_stats, 
                              distance=min_distance, 
                              height=np.mean(smoothed_stats) * 1.5)

        # 4. Selekcja 4 rzędów (układ standardowy 3x4 + 1 rytmiczny)
        if len(peaks) > 4:
            # Wybieramy piki o największej amplitudzie wariancji
            peaks = sorted(peaks, key=lambda x: smoothed_stats[x], reverse=True)[:4]
            peaks = sorted(peaks)
        elif len(peaks) < 4:
            # Fallback: jeśli algorytm zawiedzie, tniemy obraz matematycznie na 4 części
            h = binary_mask.shape[0]
            peaks = [int(h * (2*i + 1) / 8) for i in range(4)]

        # 5. Wyznaczanie granic cięcia (y_start, y_end)
        row_boundaries = []
        h = binary_mask.shape[0]
        for i in range(len(peaks)):
            # Margines bezpieczeństwa: bierzemy 90% odległości między pikami jako wysokość rzędu
            # Zapobiega to "zaciąganiu" załamków S z rzędu powyżej
            if i == 0:
                dist = peaks[i+1] - peaks[i]
                y_start = max(0, peaks[i] - dist // 2)
                y_end = peaks[i] + dist // 2
            elif i == len(peaks) - 1:
                dist = peaks[i] - peaks[i-1]
                y_start = peaks[i] - dist // 2
                y_end = min(h, peaks[i] + dist // 2)
            else:
                dist_up = peaks[i] - peaks[i-1]
                dist_down = peaks[i+1] - peaks[i]
                y_start = peaks[i] - dist_up // 2
                y_end = peaks[i] + dist_down // 2
            
            row_boundaries.append((int(y_start), int(y_end)))

        # 6. Wizualizacja kroku: step3a_row_projection.png
        if record_name:
            path = os.path.join(debug_folder, record_name, "step3a_row_projection.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            plt.figure(figsize=(8, 12))
            # Wyświetlamy surową wariancję (szary) i wygładzoną (niebieski)
            plt.plot(row_stats, np.arange(len(row_stats)), color='lightgray', label="Raw Variance")
            plt.plot(smoothed_stats, np.arange(len(smoothed_stats)), color='blue', linewidth=2, label="Smoothed")
            
            # Zaznaczamy wykryte środki rzędów
            plt.scatter(smoothed_stats[peaks], peaks, color='red', s=100, zorder=5, label="Lead Centers")
            
            # Rysujemy linie cięcia (zielone przerywane)
            for y_s, y_e in row_boundaries:
                plt.axhline(y=y_s, color='green', linestyle='--', alpha=0.6)
                plt.axhline(y=y_e, color='green', linestyle='--', alpha=0.6)

            plt.gca().invert_yaxis()
            plt.title(f"Dynamic Row Detection (Variance-based)\n{record_name}")
            plt.xlabel("Variance Magnitude")
            plt.ylabel("Pixel Y-coordinate")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()

        return row_boundaries
    
    @staticmethod
    def find_vertical_columns(row_mask_gray, num_cols=4, row_idx=0, record_name=None, debug_folder="output/debug"):
        from scipy.signal import find_peaks
        import matplotlib.pyplot as plt
        
        # 1. Przygotowanie gęstości (inwersja i wygładzanie)
        inv_gray = 255 - row_mask_gray.astype(np.float32)
        density = np.sum(inv_gray, axis=0)
        smoothed = np.convolve(density, np.ones(15)/15, mode='same')
        
        # 2. Szukamy WSZYSTKICH potencjalnych separatorów (pików gęstości)
        # Separatory to wysokie, wąskie piki.
        peaks, _ = find_peaks(smoothed, distance=200, height=np.mean(smoothed)*1.2)
        
        w = row_mask_gray.shape[1]
        
        # 3. Jeśli znaleźliśmy piki, używamy ich jako kotwic
        if len(peaks) >= 2 and (peaks[-1] - peaks[0] > w / 2):
            # Pierwszy i ostatni to nasze granice bezwzględne
            start_x = peaks[0]
            end_x = peaks[-1]
            
            # Idealne matematyczne punkty cięcia
            ideal_step = (end_x - start_x) / num_cols
            ideal_cuts = [start_x + i * ideal_step for i in range(num_cols + 1)]
            
            # SNAPPING: Dla każdego idealnego cięcia szukamy najbliższego REALNEGO piku
            final_cuts = []
            for ideal in ideal_cuts:
                # Szukamy piku w promieniu 100 pikseli od ideału
                nearby_peaks = peaks[np.abs(peaks - ideal) < 100]
                if len(nearby_peaks) > 0:
                    # Wybieramy ten, który jest najwyższy (najbardziej czarny)
                    best_peak = nearby_peaks[np.argmax(smoothed[nearby_peaks])]
                    final_cuts.append(best_peak)
                else:
                    final_cuts.append(int(ideal))
                    
            # Zabezpieczenie przed zduplikowanymi cięciami lub malejącymi:
            valid = True
            for i in range(len(final_cuts) - 1):
                if final_cuts[i+1] - final_cuts[i] < 50:
                    valid = False
                    break
            if not valid:
                step = w / num_cols
                final_cuts = [int(i * step) for i in range(num_cols + 1)]
        else:
            # Fallback jeśli zdjęcie jest zbyt zaszumione lub piki zgrupowane
            step = w / num_cols
            final_cuts = [int(i * step) for i in range(num_cols + 1)]

        col_boundaries = []
        for i in range(num_cols):
            col_boundaries.append((final_cuts[i], final_cuts[i+1]))

        # Wizualizacja - tu zobaczysz czerwone kropki na szczytach
        if record_name:
            plt.figure(figsize=(15, 3))
            plt.plot(smoothed, color='black')
            plt.scatter(peaks, smoothed[peaks], color='red', s=50) # To są nasze kotwice
            for c in final_cuts:
                plt.axvline(x=c, color='green', linewidth=2)
            plt.title(f"Anchor-Based Snapping (Row {row_idx})")
            plt.savefig(f"{debug_folder}/{record_name}/step3b_col_projection_row{row_idx}.png")
            plt.close()

        return col_boundaries

    @staticmethod
    def slice_leads_standard_3x4(row_img, lead_names_in_row, col_boundaries=None, record_name=None, debug_folder="output/debug"):
        """
        Dzieli jeden rząd na kolumny.
        lead_names_in_row: lista np. ['I', 'aVR', 'V1', 'V4']
        Zwraca słownik: {nazwa_odprowadzenia: wycięty_obraz}
        """
        w = row_img.shape[1]
        num_cols = len(lead_names_in_row)
        
        if col_boundaries is None:
            col_boundaries = [(i * w // num_cols, (i + 1) * w // num_cols) for i in range(num_cols)]
            
        segments = {}

        for i, name in enumerate(lead_names_in_row):
            start_x, end_x = col_boundaries[i]
            segment = row_img[:, start_x:end_x]
            segments[name] = segment

        return segments
