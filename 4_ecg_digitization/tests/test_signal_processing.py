import numpy as np
import os
import unittest
from src.signal_processing.converters import SignalConverter

class TestSignalProcessing(unittest.TestCase):
    def test_conversion(self):
        # Sztuczny sygnał w pikselach
        signal_px = np.array([50, 50, 40, 60, 50, np.nan, 50])
        debug_dir = "output/debug/signal_processing"
        
        signal_mv = SignalConverter.px_to_mv(signal_px, px_per_mm=10, record_name="test_rec", lead_name="Lead_dummy")
        self.assertIsNotNone(signal_mv)
        
        signal_resampled = SignalConverter.resample_to_500hz(signal_mv, current_len_mm=5, record_name="test_rec", lead_name="Lead_dummy", debug_folder=debug_dir)
        self.assertIsNotNone(signal_resampled)

if __name__ == '__main__':
    unittest.main()