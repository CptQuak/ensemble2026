import cv2
import os
import unittest
from src.preprocessing.binarization import ECGPreprocessor
from src.segmentation.segmentation import ECGSegmenter

class TestSegmentation(unittest.TestCase):
    def test_segmentation(self):
        img_path = "data/small_train/ecg_train_0048.png"
        img_bgr = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) # Dodaj skale szarości
        
        # Maska potrzebna do wykrycia rzędów (wariancja)
        mask = ECGPreprocessor.process(img_bgr, "test_rec", debug_folder="output/debug/temp")
        debug_dir = "output/debug/segmentation"
        
        row_boundaries = ECGSegmenter.find_horizontal_rows(mask, "test_rec", debug_folder=debug_dir)
        
        if len(row_boundaries) > 0:
            y_start, y_end = row_boundaries[0]
            
            # KLUCZOWA ZMIANA: Wycinamy z GRAY, nie z MASK
            row_gray = img_gray[y_start:y_end, :] 
            
            leads = ["I", "aVR", "V1", "V4"]
            
            # Teraz przekazujemy row_gray zamiast row_mask
            col_boundaries = ECGSegmenter.find_vertical_columns(row_gray, len(leads), row_idx=0, record_name="test_rec", debug_folder=debug_dir)
            self.assertTrue(os.path.exists(f"{debug_dir}/test_rec/step3b_col_projection_row0.png"))
            
            # Zamiast testować wycinki, zapisujemy na oryginalnym obrazie prostokąty z wynikami segmentacji
            grid_img = img_bgr.copy()
            for i, lead in enumerate(leads):
                x_start, x_end = col_boundaries[i]
                cv2.rectangle(grid_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)
                cv2.putText(grid_img, lead, (x_start + 10, y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            os.makedirs(f"{debug_dir}/test_rec", exist_ok=True)
            cv2.imwrite(f"{debug_dir}/test_rec/step3c_segmentation_grid.png", grid_img)
            self.assertTrue(os.path.exists(f"{debug_dir}/test_rec/step3c_segmentation_grid.png"))

if __name__ == '__main__':
    unittest.main()
