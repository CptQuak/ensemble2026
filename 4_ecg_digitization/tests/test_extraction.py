import cv2
import numpy as np
import os
import unittest
from src.extraction.baseline import BaselineExtractor

class TestExtraction(unittest.TestCase):
    def test_extraction(self):
        # Tworzenie sztucznej maski na potrzeby testu (pozioma linia)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(mask, (0, 50), (99, 50), 255, 1)
        
        debug_dir = "output/debug/extraction"
        signal = BaselineExtractor.extract_columns_full(mask, "test_rec", "Lead_dummy", debug_folder=debug_dir)
        
        self.assertEqual(len(signal), 100)
        self.assertTrue(os.path.exists(f"{debug_dir}/test_rec/step4_extracted_Lead_dummy.png"))

if __name__ == '__main__':
    unittest.main()