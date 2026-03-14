import numpy as np
import wfdb
import os

class ECGEvaluator:
    @staticmethod
    def calculate_metrics(true_signal, pred_signal):
        if len(pred_signal) == 0 or len(true_signal) == 0:
            return 0, 0, 0
        
        # Center signals
        pred_norm = pred_signal - np.mean(pred_signal)
        true_norm = true_signal - np.mean(true_signal)
        
        # Cross-correlation to find best alignment
        correlation = np.correlate(true_norm, pred_norm, mode='valid')
        if len(correlation) == 0:
            best_shift = 0
        else:
            best_shift = np.argmax(correlation)
        
        pred_aligned = pred_signal
        true_aligned = true_signal[best_shift:best_shift+len(pred_aligned)]
        
        min_len = min(len(pred_aligned), len(true_aligned))
        pred_aligned = pred_aligned[:min_len]
        true_aligned = true_aligned[:min_len]
        
        # 1. Shape (PCC) - 60 pkt
        if np.std(pred_aligned) == 0 or np.std(true_aligned) == 0:
            pcc = 0
        else:
            pcc = np.corrcoef(true_aligned, pred_aligned)[0, 1]
            if np.isnan(pcc): pcc = 0
            
        # 2. Amplitude (SNR) - 20 pkt
        noise = true_aligned - pred_aligned
        signal_power = np.sum(true_aligned**2)
        noise_power = np.sum(noise**2)
        
        if noise_power == 0:
            snr = 50.0 # perfect match
        elif signal_power == 0:
            snr = 0.0
        else:
            snr = 10 * np.log10(signal_power / noise_power)
            
        # 3. Time Calibration - 20 pkt
        # Zgodnie z wytycznymi, Time Calibration to metryka Cross-Correlation
        true_norm_aligned = true_norm[best_shift:best_shift+min_len]
        norm_factor = np.sqrt(np.sum(true_norm_aligned**2) * np.sum(pred_norm**2))
        if norm_factor == 0:
            xcorr = 0
        else:
            xcorr = correlation[best_shift] / norm_factor
            
        return max(0, pcc), snr, max(0, xcorr)

    @staticmethod
    def evaluate_submission(submission_dict, data_dir="data/small_train"):
        results = {}
        for record_name, leads_data in submission_dict.items():
            gt_path = os.path.join(data_dir, record_name)
            if not os.path.exists(gt_path + ".hea"):
                continue
                
            try:
                rec = wfdb.rdrecord(gt_path)
            except Exception as e:
                print(f"Błąd przy wczytywaniu {gt_path}: {e}")
                continue
                
            gt_signals = {}
            for i, lead in enumerate(rec.sig_name):
                gt_signals[lead] = rec.p_signal[:, i]
                
            record_results = {}
            for lead_name, pred_signal in leads_data.items():
                if lead_name in gt_signals:
                    pcc, snr, xcorr = ECGEvaluator.calculate_metrics(gt_signals[lead_name], pred_signal)
                    record_results[lead_name] = {'PCC': pcc, 'SNR': snr, 'XCorr': xcorr}
            
            if record_results:
                results[record_name] = record_results
                
        return results
        
    @staticmethod
    def print_summary(results):
        print("\n=== Raport Metryk (Task 4) ===")
        all_pcc, all_snr, all_xcorr = [], [], []
        
        for record_name, leads in results.items():
            r_pcc = np.mean([m['PCC'] for m in leads.values()])
            r_snr = np.mean([m['SNR'] for m in leads.values()])
            r_xcorr = np.mean([m['XCorr'] for m in leads.values()])
            
            all_pcc.append(r_pcc)
            all_snr.append(r_snr)
            all_xcorr.append(r_xcorr)
            
            print(f"[{record_name}] Średnie: PCC = {r_pcc:.3f}, SNR = {r_snr:.2f} dB, XCorr = {r_xcorr:.3f}")
            
        mean_pcc = np.mean(all_pcc) if all_pcc else 0
        mean_snr = np.mean(all_snr) if all_snr else 0
        mean_xcorr = np.mean(all_xcorr) if all_xcorr else 0
        
        # Obliczenie punktacji:
        # PCC - 60 pkt (proporcjonalnie do PCC)
        # SNR - 20 pkt (max za SNR > 15 dB)
        # Time Calibration - 20 pkt (proporcjonalnie do XCorr)
        
        score_pcc = mean_pcc * 60
        score_snr = min(20, max(0, (mean_snr / 15.0) * 20))
        score_time = mean_xcorr * 20
        total_score = score_pcc + score_snr + score_time
        
        print("\n--- GLOBALNE WYNIKI ---")
        print(f"Shape (PCC)      : {mean_pcc:.3f} -> {score_pcc:.1f} / 60 pkt")
        print(f"Amplitude (SNR)  : {mean_snr:.2f} dB -> {score_snr:.1f} / 20 pkt")
        print(f"Time Cal (XCorr) : {mean_xcorr:.3f} -> {score_time:.1f} / 20 pkt")
        print(f"TOTAL SCORE      : {total_score:.1f} / 100 pkt")
