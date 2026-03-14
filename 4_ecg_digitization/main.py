import cv2
import numpy as np
import os
from src.preprocessing.rotation import Deskewer
from src.preprocessing.binarization import ECGPreprocessor
from src.preprocessing.grid_detector import GridDetector
from src.preprocessing.grid_builder import GridBuilder
from src.preprocessing.homography import HomographyWarp
from src.segmentation.segmentation import ECGSegmenter
from src.extraction.baseline import BaselineExtractor
from src.utils.visualization import ECGVisualizer
from src.utils.submission import ECGSubmission
from src.signal_processing.converters import SignalConverter

def process_record(image_path, record_name, submission_obj):
    """
    Przetwarza rekord EKG: Deskew -> Homografia -> Globalna Segmentacja -> Ekstrakcja.
    """
    # 1. Wczytanie i globalne prostowanie
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Błąd: Nie można wczytać {image_path}")
        return
        
    print(f"  [{record_name}] Krok 0: Globalna korekcja rotacji...")
    img_bgr = Deskewer.fix_skew(img_bgr)

    # 2. Faza 1: Normalizacja przestrzenna (Homografia)
    print(f"  [{record_name}] Faza 1: Normalizacja przestrzenna (DLT)...")
    grid_mask = GridDetector.extract_grid_mask(img_bgr, record_name=record_name)
    vl_mask, hl_mask = GridDetector.detect_grid_lines(grid_mask, record_name=record_name)
    nodes_matrix = GridBuilder.build_grid_matrix(vl_mask, hl_mask, record_name=record_name)
    
    TARGET_PX_PER_MM = 20.0
    # W main.py wewnątrz process_record:
    img_undistorted = HomographyWarp.undistort_image_grid(
        img_bgr, nodes_matrix, cell_size_mm=1, target_px_per_mm=TARGET_PX_PER_MM
    )
    
    # Jeśli wymiary się nie zmieniły (homografia zwróciła oryginał), 
    # warto przeliczyć TARGET_PX_PER_MM dynamicznie lub założyć bezpieczny fallback.
    if img_undistorted.shape == img_bgr.shape:
        print(f"  [{record_name}] Warning: Homografia pominięta - słaba siatka.")
    
    # Praca na wyprostowanym obrazie
    img_bgr = img_undistorted
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = ECGPreprocessor.process(img_bgr, record_name)

    # 3. Segmentacja Pozioma (Rzędy)
    print(f"  [{record_name}] Faza 2: Segmentacja globalna...")
    row_boundaries = ECGSegmenter.find_horizontal_rows(mask, record_name)
    if len(row_boundaries) < 4:
        print(f"Błąd: Nie wykryto 4 rzędów w {record_name}")
        return

    # 4. Globalna Segmentacja Pionowa (Kolumny liczone RAZ)
    # Wybieramy pas obejmujący 3 górne rzędy, aby uzyskać najlepszy sygnał dla separatorów
    y_min_global = row_boundaries[0][0]
    y_max_global = row_boundaries[2][1]
    global_row_gray = img_gray[y_min_global:y_max_global, :]
    
    # Obliczamy wspólne granice kolumn dla układu 3x4
    global_col_boundaries = ECGSegmenter.find_vertical_columns(
        global_row_gray, 
        num_cols=4, 
        row_idx="GLOBAL", 
        record_name=record_name
    )

    # Przygotowanie do ekstrakcji
    grid_debug_img = img_bgr.copy()
    layout = [
        {"leads": ["I", "aVR", "V1", "V4"]},
        {"leads": ["II", "aVL", "V2", "V5"]},
        {"leads": ["III", "aVF", "V3", "V6"]},
        {"leads": ["II"]} # Rytmiczne
    ]

    for row_idx, row in enumerate(layout):
        y_start, y_end = row_boundaries[row_idx]
        leads = row["leads"]
        
        # Określenie granic kolumn dla danego rzędu
        if len(leads) > 1:
            current_cols = global_col_boundaries
        else:
            # Dla rzędu rytmicznego bierzemy pełny zakres od pierwszej do ostatniej kotwicy
            current_cols = [(global_col_boundaries[0][0], global_col_boundaries[-1][1])]
        
        row_mask = mask[y_start:y_end, :]
        row_bgr = img_bgr[y_start:y_end, :]
        
        # Segmentacja i ekstrakcja
        segments_mask = ECGSegmenter.slice_leads_standard_3x4(row_mask, leads, col_boundaries=current_cols)
        segments_bgr = ECGSegmenter.slice_leads_standard_3x4(row_bgr, leads, col_boundaries=current_cols)
        
        for idx, lead_name in enumerate(leads):
            x_start, x_end = current_cols[idx]
            
            # Debug: Rysowanie siatki
            cv2.rectangle(grid_debug_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(grid_debug_img, lead_name, (x_start + 5, y_start + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Ekstrakcja i konwersja
            roi_mask = segments_mask[lead_name]
            signal_px_raw = BaselineExtractor.extract_columns_full(roi_mask, record_name, lead_name)
            
            # Overlay diagnostyczny
            img_roi = segments_bgr[lead_name]
            diag_overlay = ECGVisualizer.create_overlay(img_roi, signal_px_raw)
            ECGVisualizer.save_debug_image(diag_overlay, f"output/debug/{record_name}/overlay_{lead_name}.png")
            
            # Kalibracja (1mm = 20px, 10mm/mV)
            signal_mv = SignalConverter.px_to_mv(signal_px_raw, TARGET_PX_PER_MM, record_name=record_name, lead_name=lead_name)
            
            # Resampling do 500Hz
            len_mm = len(signal_px_raw) / TARGET_PX_PER_MM
            signal_final = SignalConverter.resample_to_500hz(signal_mv, len_mm, record_name=record_name, lead_name=lead_name)
            
            submission_obj.add_lead(record_name, lead_name, signal_final)

    ECGVisualizer.save_debug_image(grid_debug_img, f"output/debug/{record_name}/step3c_segmentation_grid.png")

def main():
    submission = ECGSubmission()
    data_dir = "data/small_train"
    
    # Lista plików do przetworzenia
    files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    
    for record_file in files:
        record_id = record_file.replace('.png', '')
        print(f"\n>>> Rozpoczynam pracę nad: {record_id}")
        process_record(f'{data_dir}/{record_file}', record_id, submission)
    
    submission.save("output/submission.npz")
    print("\n[SUKCES] Potok przetwarzania zakończony.")

if __name__ == "__main__":
    main()