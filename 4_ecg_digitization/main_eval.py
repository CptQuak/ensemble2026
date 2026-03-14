import cv2
import numpy as np
import os
import glob
from main import process_record
from src.utils.submission import ECGSubmission
from src.utils.evaluation import ECGEvaluator

def main():
    submission = ECGSubmission()
    data_dir = "data/small_train"
    
    # Pobierz listę wszystkich plików PNG w katalogu small_train
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    
    if not image_paths:
        print(f"Błąd: Nie znaleziono plików .png w {data_dir}")
        return

    print(f"Znaleziono {len(image_paths)} rekordów do przetworzenia.")

    for img_path in image_paths:
        # Wyciągamy nazwę rekordu z nazwy pliku (np. ecg_train_0001)
        record_name = os.path.splitext(os.path.basename(img_path))[0]
        
        print(f"\n>>> Przetwarzanie rekordu: {record_name}")
        try:
            process_record(img_path, record_name, submission)
        except Exception as e:
            print(f"Błąd podczas przetwarzania {record_name}: {e}")

    # Konwersja płaskiego słownika submission na zagnieżdżony dla ewaluatora
    # submission.predictions: {"ecg_train_0001_I": signal, ...}
    nested_preds = {}
    for key, signal in submission.predictions.items():
        # Szukamy ostatniego podkreślenia, aby oddzielić lead_name
        record_id, lead_name = key.rsplit('_', 1)
        if record_id not in nested_preds:
            nested_preds[record_id] = {}
        nested_preds[record_id][lead_name] = signal

    # Uruchomienie ewaluacji
    print("\n" + "="*50)
    print("URUCHAMIANIE EWALUACJI METRYK")
    print("="*50)
    
    results = ECGEvaluator.evaluate_submission(nested_preds, data_dir=data_dir)
    ECGEvaluator.print_summary(results)
    
    # Opcjonalnie zapisz wynik submission
    submission.save("output/submission_small_train.npz")

if __name__ == "__main__":
    main()
