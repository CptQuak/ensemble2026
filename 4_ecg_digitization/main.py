import cv2
import numpy as np
import os
from src.preprocessing.rotation import Deskewer
from src.preprocessing.binarization import ECGPreprocessor
from src.preprocessing.grid_detector import GridDetector
from src.preprocessing.grid_builder import GridBuilder
from src.preprocessing.homography import HomographyWarp
from src.segmentation.segmentation import ECGSegmenter
from src.extraction.viterbi import ViterbiExtractor
from src.utils.visualization import ECGVisualizer
from src.utils.submission import ECGSubmission
from src.signal_processing.converters import SignalConverter

def process_record(image_path, record_name, submission_obj):
    """
    Przetwarza rekord EKG: Deskew -> Homografia -> Globalna Segmentacja -> Ekstrakcja Viterbi -> Anchoring.
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
    img_undistorted = HomographyWarp.undistort_image_grid(
        img_bgr, nodes_matrix, cell_size_mm=1, target_px_per_mm=TARGET_PX_PER_MM
    )
    
    if img_undistorted.shape == img_bgr.shape:
        print(f"  [{record_name}] Warning: Homografia pominięta - słaba siatka.")
    
    img_bgr = img_undistorted
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Rozdzielenie na dwie strategie binaryzacji
    mask_segmentation = ECGPreprocessor.get_segmentation_mask(img_bgr, record_name)
    mask_extraction, gray_no_grid = ECGPreprocessor.get_extraction_mask(img_bgr, record_name)

    # 3. Segmentacja Pozioma (Rzędy)
    print(f"  [{record_name}] Faza 2: Segmentacja globalna...")
    row_boundaries = ECGSegmenter.find_horizontal_rows(mask_segmentation, record_name)
    if len(row_boundaries) < 4:
        print(f"Błąd: Nie wykryto 4 rzędów w {record_name}")
        return

    # 4. Globalna Segmentacja Pionowa
    y_min_global = row_boundaries[0][0]
    y_max_global = row_boundaries[2][1]
    global_row_gray = img_gray[y_min_global:y_max_global, :]
    
    global_col_boundaries = ECGSegmenter.find_vertical_columns(
        global_row_gray, 
        num_cols=4, 
        row_idx="GLOBAL", 
        record_name=record_name
    )

    grid_debug_img = img_bgr.copy()
    layout = [
        {"leads": ["I", "aVR", "V1", "V4"]},
        {"leads": ["II", "aVL", "V2", "V5"]},
        {"leads": ["III", "aVF", "V3", "V6"]},
        {"leads": ["II"]} # Rytmiczne
    ]

    print(f"  [{record_name}] Faza 3 i 4: Ekstrakcja sygnału (Viterbi) i Anchoring...")
    for row_idx, row in enumerate(layout):
        y_start, y_end = row_boundaries[row_idx]
        leads = row["leads"]
        
        if len(leads) > 1:
            current_cols = global_col_boundaries
        else:
            current_cols = [(global_col_boundaries[0][0], global_col_boundaries[-1][1])]
        
        row_bgr = img_bgr[y_start:y_end, :]
        row_mask_ext = mask_extraction[y_start:y_end, :]
        row_gray_ext = gray_no_grid[y_start:y_end, :] 
        
        segments_mask = ECGSegmenter.slice_leads_standard_3x4(row_mask_ext, leads, col_boundaries=current_cols)
        segments_bgr = ECGSegmenter.slice_leads_standard_3x4(row_bgr, leads, col_boundaries=current_cols)
        segments_gray = ECGSegmenter.slice_leads_standard_3x4(row_gray_ext, leads, col_boundaries=current_cols)
        
        for idx, lead_name in enumerate(leads):
            x_start, x_end = current_cols[idx]
            
            cv2.rectangle(grid_debug_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(grid_debug_img, lead_name, (x_start + 5, y_start + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # 5. Ekstrakcja danych algorytmem Viterbiego
            margin = 10 
            roi_mask_trimmed = segments_mask[lead_name][:, margin:-margin]
            roi_gray_trimmed = segments_gray[lead_name][:, margin:-margin]
            
            signal_px_trimmed = ViterbiExtractor.extract_signal(
                roi_mask_trimmed, 
                roi_gray_trimmed, 
                record_name=record_name, 
                lead_name=lead_name
            )
            
            signal_px_raw = np.pad(signal_px_trimmed, (margin, margin), mode='edge')
            
            # Overlay diagnostyczny
            img_roi = segments_bgr[lead_name]
            diag_overlay = ECGVisualizer.create_overlay(img_roi, signal_px_raw)
            ECGVisualizer.save_debug_image(diag_overlay, f"output/debug/{record_name}/overlay_{lead_name}.png")
            
            # 6. Faza 4: Kalibracja i Resampling z Anchoringiem (Kompensacja Fazy)
            signal_mv = SignalConverter.px_to_mv(signal_px_raw, TARGET_PX_PER_MM)
            
            # Przekazujemy x_start (globalną pozycję na obrazie), aby usunąć Temporal Shift
            signal_final = SignalConverter.resample_to_500hz_anchored(
                signal_mv, 
                start_x_global=x_start, 
                target_px_per_mm=TARGET_PX_PER_MM,
                record_name=record_name, 
                lead_name=lead_name
            )
            
            submission_obj.add_lead(record_name, lead_name, signal_final)

    ECGVisualizer.save_debug_image(grid_debug_img, f"output/debug/{record_name}/step3c_segmentation_grid.png")

def main():
    submission = ECGSubmission()
    data_dir = "data/small_train"
    
    # Procesujemy testowy rekord
    process_record(f'{data_dir}/ecg_train_0008.png', '0008XD', submission)
    
    submission.save("output/submission.npz")
    print("\n[SUKCES] Potok przetwarzania zakończony.")

if __name__ == "__main__":
    main()