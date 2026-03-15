import os, glob, requests
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from main import process_record
from src.utils.submission import ECGSubmission

def process_single_image(img_path):
    record_name = os.path.splitext(os.path.basename(img_path))[0]
    local_sub = ECGSubmission()
    try:
        process_record(img_path, record_name, local_sub)
        return record_name, local_sub.predictions, None
    except Exception as e:
        import traceback
        return record_name, None, traceback.format_exc()

def main():
    final_submission = ECGSubmission()
    data_dir = "data/test"
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    
    print(f"Start: Przetwarzanie {len(image_paths)} obrazów...")

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_image, p): p for p in image_paths}
        for future in tqdm(as_completed(futures), total=len(image_paths)):
            name, preds, err = future.result()
            if preds:
                final_submission.predictions.update(preds)
            elif err:
                print(f"Błąd {name}: {err}")

    final_submission.save("output/submission.npz")
    print("Sukces: Plik submission.npz wygenerowany.")

if __name__ == "__main__":
    main()