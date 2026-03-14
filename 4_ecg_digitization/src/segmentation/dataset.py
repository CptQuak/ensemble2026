import os
import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from .augmentations import AdvancedECGAugmenter

class ECGDataset(Dataset):
    """
    Klasa Dataset dla modelu ResNet34 U-Net ze specyficznym reżimem augmentacji.
    """
    def __init__(self, data_dir, is_train=True, target_size=(512, 512)):
        self.data_dir = data_dir
        self.is_train = is_train
        self.target_size = target_size
        
        # Wyszukaj wszystkie pliki PNG
        self.image_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        
        # Inicjalizacja augmentera, który wykonuje OverlaySignal + AddVerticalLines
        self.augmenter = AdvancedECGAugmenter(data_dir=data_dir) if is_train else None

    def __len__(self):
        return len(self.image_files)

    def _get_dummy_mask(self, image):
        """
        Z uwagi na to, że prawdziwe maski (Ground Truth) EKG zazwyczaj trzeba 
        najpierw wygenerować klasycznie (pseudolabeling) lub pobrać z 
        zewnętrznego źródła, na etapie inicjalizacji generujemy maskę zastępczą
        za pomocą naszej starej metody Otsu (ECGPreprocessor), 
        aby pipeline uczący w ogóle mógł ruszyć.
        """
        from src.preprocessing.binarization import ECGPreprocessor
        
        # Zastępcze ground truth
        mask, _ = ECGPreprocessor.get_extraction_mask(image, record_name="dummy_train")
        
        # Konwersja na format 0..1 i wymiary (H, W, 1)
        mask = (mask > 0).astype(np.float32)
        return mask

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Wczytanie obrazka BGR
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Nie udało się wczytać: {img_path}")

        # Jeśli to zbiór uczący, aplikujemy Adversarial Overlap (p=0.4) i Linie (p=0.3)
        if self.is_train and self.augmenter:
            image = self.augmenter(image)

        # Na potrzeby treningu musimy mieć labelkę.
        # UWAGA: W prawdziwym środowisku wczytalibyśmy maskę ręcznie zrobioną
        # lub wygenerowaną przez algorytmy Viterbiego.
        mask = self._get_dummy_mask(image)

        # Zmiana rozmiaru do jednolitego dla sieci
        image_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Normalizacja do [0, 1] dla modelu
        image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)

        return image_tensor, mask_tensor
