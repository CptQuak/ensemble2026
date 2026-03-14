import os
import glob
import cv2
import numpy as np
import requests
import matplotlib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv

# Ustawienie backendu matplotlib na Agg przed importem main/submission
matplotlib.use('Agg')

from main import process_record
from src.utils.submission import ECGSubmission

# Załaduj zmienne środowiskowe (TEAM_TOKEN, SERVER_URL)
load_dotenv()

ENDPOINT = "task4"
API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")
NPZ_FILE = "output/submission.npz"

def process_single_image(img_path):
    """Przetwarza pojedynczy obraz i zwraca wyniki."""
    record_name = os.path.splitext(os.path.basename(img_path))[0]
    local_submission = ECGSubmission()
    try:
        process_record(img_path, record_name, local_submission)
        return record_name, local_submission.predictions, None
    except Exception as e:
        import traceback
        return record_name, None, traceback.format_exc()

def main():
    submission = ECGSubmission()
    data_dir = "data/test"
    
    # 1. Pobierz listę wszystkich plików PNG w katalogu test
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    
    if not image_paths:
        print(f"Błąd: Nie znaleziono plików .png w {data_dir}")
        return

    print(f"Znaleziono {len(image_paths)} rekordów do przetworzenia w zbiorze TEST.")

    # 2. Procesuj rekordy równolegle
    num_workers = os.cpu_count()
    print(f"Uruchamiam przetwarzanie na {num_workers} rdzeniach...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Mapowanie ścieżek na taski
        future_to_path = {executor.submit(process_single_image, path): path for path in image_paths}
        
        # Pasek postępu tqdm
        with tqdm(total=len(image_paths), desc="Processing ECGs") as pbar:
            for future in as_completed(future_to_path):
                record_name, predictions, error = future.result()
                if error:
                    print(f"\n  !! Błąd w {record_name}:\n{error}")
                elif predictions:
                    submission.predictions.update(predictions)
                pbar.update(1)

    # 3. Zapisz wyniki do NPZ (zgodnie ze specyfikacją Task 4)
    if submission.predictions:
        submission.save(NPZ_FILE)
    else:
        print("Błąd: Nie zebrano żadnych wyników. Pomijam zapis.")
        return

    # 4. Wyślij na serwer (jeśli skonfigurowano)
    if not API_TOKEN or not SERVER_URL:
        print("\n[INFO] TEAM_TOKEN lub SERVER_URL nie są zdefiniowane w .env. Pomijam wysyłkę.")
        print(f"Plik submission jest gotowy w: {NPZ_FILE}")
        return

    print(f"\n[UPLOAD] Wysyłanie {NPZ_FILE} na serwer {SERVER_URL}...")
    headers = {"X-API-Token": API_TOKEN}
    
    try:
        with open(NPZ_FILE, "rb") as f:
            response = requests.post(
                f"{SERVER_URL}/{ENDPOINT}",
                files={"npz_file": f},
                headers=headers
            )
        
        try:
            data = response.json()
        except Exception:
            data = response.text
            
        print(f"Odpowiedź serwera ({response.status_code}):", data)
    except Exception as e:
        print(f"Błąd podczas wysyłki: {e}")

if __name__ == "__main__":
    main()
