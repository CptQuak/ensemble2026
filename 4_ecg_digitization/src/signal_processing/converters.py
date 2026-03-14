import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os

class SignalConverter:
    """
    Moduł Inżynierii Sygnału odpowiadający za Faze 4 i 5 potoku.
    Implementuje Grid Anchoring, resampling do 500 Hz oraz filtrację zero-phase.
    """
    
    @staticmethod
    def save_debug_image(signal, title, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure(figsize=(10, 2))
        plt.plot(signal, color='green', linewidth=1)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    @staticmethod
    def px_to_mv(signal_px, px_per_mm, gain=10):
        """
        Konwertuje sygnał z domeny pikseli na miliwolty.
        Gwarantuje poprawność metryki Amplitude (SNR). [cite: 26]
        Standard: 10 mm/mV. [cite: 135]
        """
        # Obliczenie odwróconego sygnału (piksele rosną w dół)
        signal_inv = -signal_px 
        signal_mv = signal_inv / (px_per_mm * gain)
        return signal_mv

    @staticmethod
    def apply_butterworth_filter(signal_500hz, cutoff=150.0, fs=500.0, order=4):
        """
        Aplikuje dolnoprzepustowy filtr Butterwortha 4. rzędu. [cite: 181]
        Używa filtfilt (zero-phase), aby uniknąć opóźnień grupowych. [cite: 182]
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal_500hz)

    @staticmethod
    def resample_to_500hz_anchored(signal_mv, start_x_global, target_px_per_mm=20.0, 
                                  record_name=None, lead_name=None, debug_folder="output/debug"):
        """
        Faza 4: Grid Anchoring - synchronizacja fazy sygnału z siatką. [cite: 166]
        Eliminuje Temporal Shift w metryce Cross-Correlation. [cite: 157, 171]
        """
        num_pixels = len(signal_mv)
        current_len_mm = num_pixels / target_px_per_mm
        
        # 1. Obliczenie docelowej liczby próbek (dokładnie 20 próbek na 1 mm) [cite: 162]
        target_samples = int(np.round(current_len_mm * 20))
        if target_samples <= 1:
            target_samples = 2
            
        # 2. Obliczenie błędu fazowego względem fizycznej siatki milimetrowej 
        # Sprawdzamy przesunięcie punktu cięcia względem siatki 1mm (target_px_per_mm)
        phase_shift_px = start_x_global % target_px_per_mm
        
        # Przeliczenie błędu na ułamkowe przesunięcie w milimetrach 
        fractional_shift_mm = phase_shift_px / target_px_per_mm
            
        # 3. Interpolacja Cubic Spline z kompensacją przesunięcia [cite: 169]
        # x_new jest przesunięte tak, aby siatka uderzała w indeksy podzielne przez 20 
        x_old = np.linspace(0, current_len_mm, num_pixels)
        x_new = np.linspace(0, current_len_mm, target_samples) - fractional_shift_mm
        
        f = interp1d(x_old, signal_mv, kind='cubic', fill_value="extrapolate")
        resampled = f(x_new)
        
        # 4. Filtracja 150 Hz (Faza 5) [cite: 181]
        filtered = SignalConverter.apply_butterworth_filter(resampled)
        
        # 5. Baseline Wander Removal - odjęcie mediany [cite: 183]
        final_signal = filtered - np.median(filtered)
        
        # 6. Konwersja do float16 (Wymóg optymalizacji I/O) 
        final_signal = final_signal.astype(np.float16)
        
        if record_name and lead_name:
            path = os.path.join(debug_folder, record_name, f"step6_final_anchored_{lead_name}.png")
            SignalConverter.save_debug_image(final_signal, f"Final Anchored (500Hz, mV) - {lead_name}", path)
            
        return final_signal