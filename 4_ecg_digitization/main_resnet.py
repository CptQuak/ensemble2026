import cv2
import numpy as np
import os
import torch

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
from src.segmentation.unet_resnet34 import ResNet34UNet

def process_record(image_path, record_name, submission_obj, model=None, device=None):
    """
    Przetwarza rekord EKG: Deskew -> Homografia -> Segmentacja ResNet -> Ekstrakcja Viterbi.
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
    
    # Zabezpieczenie przed błędem homografii (fallback do oryginału)
    if img_undistorted.shape == img_bgr.shape:
        print(f"  [{record_name}] Warning: Homografia pominięta - słaba siatka.")
    
    # Praca na wyprostowanym obrazie
    img_bgr = img_undistorted
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    if model is not None and device is not None:
        print(f"  [{record_name}] Faza 1.5: Generowanie maski segmentacji przy pomocy ResNet-UNet...")
        h, w = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            output = model(img_tensor)
            output_prob = torch.sigmoid(output).squeeze().cpu().numpy()
            
        mask_resnet = cv2.resize(output_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_resnet_bin = (mask_resnet > 0.5).astype(np.uint8) * 255
        
        # Zapisz debug maski ResNet
        if os.environ.get("DEBUG_MODE", "0") == "1":
            os.makedirs(f"output/debug/{record_name}", exist_ok=True)
            cv2.imwrite(f"output/debug/{record_name}/step2_resnet_mask.png", mask_resnet_bin)

        mask_segmentation = mask_resnet_bin
        mask_extraction = mask_resnet_bin
        
        # Mapa luminancji dla Viterbi (bez siatki)
        b, g, r = cv2.split(img_bgr)
        gray_no_grid = cv2.max(cv2.max(b, g), r)
    else:
        # Fallback na stare rozwiązanie
        mask_segmentation = ECGPreprocessor.get_segmentation_mask(img_bgr, record_name)
        mask_extraction, gray_no_grid = ECGPreprocessor.get_extraction_mask(img_bgr, record_name)

    # 3. Segmentacja Pozioma (Rzędy) - używa maski segmentacyjnej
    print(f"  [{record_name}] Faza 2: Segmentacja globalna...")
    row_boundaries = ECGSegmenter.find_horizontal_rows(mask_segmentation, record_name)
    if len(row_boundaries) < 4:
        print(f"Błąd: Nie wykryto 4 rzędów w {record_name}")
        return

    # 4. Globalna Segmentacja Pionowa
    # Wybieramy 3 główne rzędy, by precyzyjniej znaleźć separatory
    y_min_global = row_boundaries[0][0]
    y_max_global = row_boundaries[2][1]
    global_row_gray = img_gray[y_min_global:y_max_global, :]
    
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

    print(f"  [{record_name}] Faza 3: Ekstrakcja sygnału (Viterbi)...")
    for row_idx, row in enumerate(layout):
        y_start, y_end = row_boundaries[row_idx]
        leads = row["leads"]
        
        # Ostatni rząd (rytmiczny) bierzemy w całości
        if len(leads) > 1:
            current_cols = global_col_boundaries
        else:
            current_cols = [(global_col_boundaries[0][0], global_col_boundaries[-1][1])]
        
        row_bgr = img_bgr[y_start:y_end, :]
        # Dla Viterbiego wycinamy maskę ekstrakcyjną i czystą mapę luminancji
        row_mask_ext = mask_extraction[y_start:y_end, :]
        row_gray_ext = gray_no_grid[y_start:y_end, :] 
        
        segments_mask = ECGSegmenter.slice_leads_standard_3x4(row_mask_ext, leads, col_boundaries=current_cols)
        segments_bgr = ECGSegmenter.slice_leads_standard_3x4(row_bgr, leads, col_boundaries=current_cols)
        segments_gray = ECGSegmenter.slice_leads_standard_3x4(row_gray_ext, leads, col_boundaries=current_cols)
        
        for idx, lead_name in enumerate(leads):
            x_start, x_end = current_cols[idx]
            
            # Debug: Rysowanie siatki
            cv2.rectangle(grid_debug_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(grid_debug_img, lead_name, (x_start + 5, y_start + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # 5. Ekstrakcja danych algorytmem Viterbiego
            # Odcinamy 10 pikseli z lewej i prawej strony, aby ominąć grube czarne separatory i napisy
            margin = 10 
            roi_mask_trimmed = segments_mask[lead_name][:, margin:-margin]
            roi_gray_trimmed = segments_gray[lead_name][:, margin:-margin]
            
            signal_px_trimmed = ViterbiExtractor.extract_signal(
                roi_mask_trimmed, 
                roi_gray_trimmed, 
                record_name=record_name, 
                lead_name=lead_name
            )
            
            # Odbudowujemy pełną długość sygnału (uzupełniamy obcięte marginesy pierwszą/ostatnią wartością)
            signal_px_raw = np.pad(signal_px_trimmed, (margin, margin), mode='edge')
            
            # Overlay diagnostyczny (sprawdzenie czy Viterbi trzyma się sygnału)
            img_roi = segments_bgr[lead_name]
            diag_overlay = ECGVisualizer.create_overlay(img_roi, signal_px_raw)
            ECGVisualizer.save_debug_image(diag_overlay, f"output/debug/{record_name}/overlay_{lead_name}.png")
            
            # 6. Kalibracja i Resampling
            signal_mv = SignalConverter.px_to_mv(signal_px_raw, TARGET_PX_PER_MM)

            len_mm = len(signal_px_raw) / TARGET_PX_PER_MM
            # Replace resample_to_500hz with resample_to_500hz_anchored because resample_to_500hz doesn't exist.
            signal_final = SignalConverter.resample_to_500hz_anchored(
                signal_mv,
                start_x_global=x_start,
                is_rhythm_strip=(len(leads) == 1),
                target_px_per_mm=TARGET_PX_PER_MM
            )            
            submission_obj.add_lead(record_name, lead_name, signal_final)

    ECGVisualizer.save_debug_image(grid_debug_img, f"output/debug/{record_name}/step3c_segmentation_grid.png")

from src.utils.evaluation import ECGEvaluator
import glob

def main():
    submission = ECGSubmission()
    data_dir = "data/small_train"

    print("[ResNet] Inicjalizacja modelu U-Net ResNet34...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet34UNet(in_channels=3, out_channels=1, pretrained=True).to(device)

    # Ładowanie wag, jeśli istnieją, w przeciwnym razie domyślne.
    weights_path = "models/unet_resnet34_best.pth"
    if os.path.exists(weights_path):
        print(f"[ResNet] Wczytywanie wytrenowanych wag z {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"[ResNet] Brak pliku z wagami ({weights_path}). Używam wag domyślnych (ImageNet/inicjalizacja).")

    model.eval()

    # Przetwarzanie całego zbioru small_train
    image_paths = sorted(glob.glob(f"{data_dir}/*.png"))

    for img_path in image_paths:
        record_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n--- Przetwarzanie: {record_name} ---")
        process_record(img_path, record_name, submission, model=model, device=device)

    submission.save("output/submission_resnet.npz")

    # Ewaluacja metryk
    print("\n[Ewaluacja] Rozpoczynam ocenę na podstawie Ground Truth...")
    results = ECGEvaluator.evaluate_submission(submission.signals, data_dir=data_dir)
    ECGEvaluator.print_summary(results)

    print("\n[SUKCES] Potok przetwarzania zakończony.")

if __name__ == "__main__":
    main()