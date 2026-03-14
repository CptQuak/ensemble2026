import cv2
import os
import unittest
from src.preprocessing.binarization import ECGPreprocessor

class TestPreprocessing(unittest.TestCase):
    def test_process(self):
        img_path = "data/small_train/ecg_train_0048.png"
        img_bgr = cv2.imread(img_path)
        self.assertIsNotNone(img_bgr, "Nie udało się wczytać obrazu testowego")
        
        # Test przetwarzania i zapisu do wyznaczonego folderu debug
        debug_dir = "output/debug/preprocessing"
        mask = ECGPreprocessor.get_segmentation_mask(img_bgr, "test_rec", debug_folder=debug_dir)
        
        self.assertIsNotNone(mask)
        self.assertTrue(os.path.exists(f"{debug_dir}/test_rec/step2_segmentation_mask.png"))

if __name__ == '__main__':
    unittest.main()