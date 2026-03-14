import numpy as np
import os
import matplotlib.pyplot as plt

class ECGSubmission:
    """Moduł odpowiedzialny za finalne formatowanie i zapis wyników Task 4."""

    def __init__(self):
        self.predictions = {}

    @staticmethod
    def save_debug_image(signals_dict, record_name, path):
        """Zapisuje wizualizację finalnych sygnałów z submission jako .png."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        prefix = f"{record_name}_"
        leads = [k[len(prefix):] for k in signals_dict.keys() if k.startswith(prefix)]
        if not leads:
            return
            
        fig, axes = plt.subplots(len(leads), 1, figsize=(10, 2 * len(leads)))
        if len(leads) == 1:
            axes = [axes]
            
        for ax, lead in zip(axes, leads):
            key = f"{record_name}_{lead}"
            ax.plot(signals_dict[key], color='green')
            ax.set_title(f"Final Signal - {lead}")
            
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def add_lead(self, record_name, lead_name, signal):
        """
        Dodaje odczytany sygnał do słownika submission.
        signal: 1D numpy array (float16, 500Hz).
        """
        key = f"{record_name}_{lead_name}"
        self.predictions[key] = signal.astype(np.float16)

    def save(self, output_path="output/submission.npz", debug_folder="output/debug"):
        """Zapisuje skompresowane archiwum .npz oraz generuje podsumowujące PNG."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, **self.predictions)
        print(f"Pomyślnie zapisano submission: {output_path}")
        print(f"Liczba zapisanych kanałów: {len(self.predictions)}")
        
        records = set([k.rsplit('_', 1)[0] for k in self.predictions.keys()])
        for rec in records:
            path = os.path.join(debug_folder, rec, "step7_final_submission.png")
            self.save_debug_image(self.predictions, rec, path)