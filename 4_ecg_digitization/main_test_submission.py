import os
import glob
import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from main import process_record
from src.utils.submission import ECGSubmission

# Załaduj zmienne środowiskowe (TEAM_TOKEN, SERVER_URL)
load_dotenv()

ENDPOINT = "task4"
API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")
NPZ_FILE = "output/submission.npz"

def main():
    submission = ECGSubmission()
    data_dir = "data/test"
    
    # 1. Pobierz listę wszystkich plików PNG w katalogu test
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    
    if not image_paths:
        print(f"Błąd: Nie znaleziono plików .png w {data_dir}")
        return

    print(f"Znaleziono {len(image_paths)} rekordów do przetworzenia w zbiorze TEST.")

    # 2. Procesuj każdy rekord
    for i, img_path in enumerate(image_paths):
        record_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{i+1}/{len(image_paths)}] Przetwarzanie: {record_name}")
        try:
            process_record(img_path, record_name, submission)
        except Exception as e:
            print(f"  !! Błąd w {record_name}: {e}")

    # 3. Zapisz wyniki do NPZ (zgodnie ze specyfikacją Task 4)
    submission.save(NPZ_FILE)

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
