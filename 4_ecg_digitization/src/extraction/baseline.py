import numpy as np
import matplotlib.pyplot as plt
import os

class BaselineExtractor:
    """Moduł do ekstrakcji bazowego sygnału pikselowego."""
    
    @staticmethod
    def save_debug_image(signal_px, path):
        """Zapisuje wykres wyekstrahowanego sygnału jako .png."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure(figsize=(10, 2))
        plt.plot(signal_px, color='black')
        plt.title("Extracted Signal (px)")
        plt.gca().invert_yaxis()
        plt.savefig(path)
        plt.close()

    @staticmethod
    def extract_columns_full(binary_mask, record_name=None, lead_name=None, debug_folder="output/debug"):
        """
        KROK 1: Algorytm uśredniający pozycje aktywnych pikseli.
        binary_mask: Maska po progowaniu Otsu (sygnał = 255).
        """
        height, width = binary_mask.shape
        signal_px = np.zeros(width)
        
        for x in range(width):
            column = binary_mask[:, x]
            active_px = np.where(column == 255)[0]
            
            if len(active_px) > 0:
                # Full Extraction: średnia arytmetyczna
                signal_px[x] = np.mean(active_px)
            else:
                signal_px[x] = np.nan
        
        if record_name and lead_name:
            path = os.path.join(debug_folder, record_name, f"step4_extracted_{lead_name}.png")
            BaselineExtractor.save_debug_image(signal_px, path)
            
        return signal_px