import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

class SignalConverter:
    """Moduł do konwersji jednostek sygnału."""
    
    @staticmethod
    def save_debug_image(signal, title, path):
        """Zapisuje wykres przekonwertowanego sygnału jako .png."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure(figsize=(10, 2))
        plt.plot(signal, color='blue')
        plt.title(title)
        plt.savefig(path)
        plt.close()

    @staticmethod
    def px_to_mv(signal_px, px_per_mm, gain=10, record_name=None, lead_name=None, debug_folder="output/debug"):
        """Konwersja pikseli na miliwolty (Standard: 10mm/mV)."""
        valid_px = signal_px[~np.isnan(signal_px)]
        median_val = np.median(valid_px) if len(valid_px) > 0 else 0
        centered = median_val - signal_px
        signal_mv = centered / (px_per_mm * gain)
        
        if record_name and lead_name:
            path = os.path.join(debug_folder, record_name, f"step5_mv_{lead_name}.png")
            SignalConverter.save_debug_image(signal_mv, f"Signal in mV - {lead_name}", path)
            
        return signal_mv

    @staticmethod
    def resample_to_500hz(signal_mv, current_len_mm, record_name=None, lead_name=None, debug_folder="output/debug"):
        """Resampling do 500Hz: dokładnie 20 próbek na 1mm."""
        nan_mask = np.isnan(signal_mv)
        if np.any(nan_mask):
            valid_idx = np.where(~nan_mask)[0]
            if len(valid_idx) > 0:
                f_nan = interp1d(valid_idx, signal_mv[valid_idx], kind='linear', fill_value="extrapolate")
                signal_mv = f_nan(np.arange(len(signal_mv)))
            else:
                signal_mv = np.zeros_like(signal_mv)

        target_samples = int(current_len_mm * 20)
        if target_samples <= 1:
            target_samples = 2
            
        f = interp1d(np.linspace(0, 1, len(signal_mv)), signal_mv, kind='cubic')
        resampled = f(np.linspace(0, 1, target_samples))
        resampled = resampled.astype(np.float16)
        
        if record_name and lead_name:
            path = os.path.join(debug_folder, record_name, f"step6_resampled_{lead_name}.png")
            SignalConverter.save_debug_image(resampled, f"Resampled (500Hz) - {lead_name}", path)
            
        return resampled