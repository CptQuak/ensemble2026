import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

class SignalConverter:
    @staticmethod
    def px_to_mv(signal_px, px_per_mm, gain=10):
        """Konwersja pikseli na mV zgodnie ze standardem 10mm/mV."""
        # Inwersja osi Y (piksele na obrazie rosną w dół)
        return (-signal_px) / (px_per_mm * gain)

    @staticmethod
    def apply_butterworth_filter(signal, cutoff=150.0, fs=500.0):
        """Filtr dolnoprzepustowy Butterwortha 4. rzędu (zero-phase)."""
        nyq = 0.5 * fs
        b, a = butter(4, cutoff / nyq, btype='low')
        if len(signal) <= max(len(a), len(b)) * 3:
            return signal
        return filtfilt(b, a, signal)

    @staticmethod
    def resample_to_500hz_anchored(signal_mv, start_x_global, is_rhythm_strip=False, 
                                  target_px_per_mm=20.0):
        """
        Faza 4: Grid Anchoring - synchronizacja fazy sygnału z fizyczną siatką.
        Wymusza stałą długość zapisu dla 10-sekundowego Ground Truth.
        """
        # Krótkie odprowadzenia: 2.5s (1250 próbek), Pasek rytmiczny: 10s (5000 próbek)
        target_len = 5000 if is_rhythm_strip else 1250
        num_pixels = len(signal_mv)
        
        if num_pixels < 2:
            return np.zeros(target_len, dtype=np.float16)

        # Obliczenie błędu fazowego względem fizycznej siatki 1mm
        # Sprawdzamy przesunięcie punktu startowego względem cyklu siatki milimetrowej
        phase_shift_px = start_x_global % target_px_per_mm
        fractional_shift_mm = phase_shift_px / target_px_per_mm
        
        # Interpolacja Cubic Spline na docelową siatkę czasu z kompensacją przesunięcia
        len_mm = num_pixels / target_px_per_mm
        x_old = np.linspace(0, len_mm, num_pixels)
        x_new = np.linspace(0, len_mm, target_len) - fractional_shift_mm
        
        f = interp1d(x_old, signal_mv, kind='cubic', fill_value="extrapolate")
        resampled = f(x_new)
        
        # Usuwanie szumu wysokoczęstotliwościowego i stałej składowej
        filtered = SignalConverter.apply_butterworth_filter(resampled)
        final_signal = filtered - np.median(filtered)
        
        # Rzutowanie do float16 zgodnie ze specyfikacją I/O
        return final_signal.astype(np.float16)