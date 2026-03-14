import cv2
import numpy as np
import random
import glob
import os

class AdvancedECGAugmenter:
    """
    Krytyczne augmentacje wymienione w strategii (Zadanie 4):
    OverlaySignal (nałożenie innego fragmentu ekg) + Szum "wyblakłego tekstu"
    """
    def __init__(self, data_dir="data/small_train"):
        self.data_dir = data_dir
        self.image_files = []
        if os.path.exists(data_dir):
            self.image_files = glob.glob(os.path.join(data_dir, "*.png"))

    def overlay_signal(self, image, probability=0.4):
        """
        P_overlay = 0.4.
        Wstrzykuje losowy fragment z innej próbki treningowej do obrazu bazowego.
        """
        if random.random() > probability or not self.image_files:
            return image
            
        bg_path = random.choice(self.image_files)
        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            return image
            
        h, w = image.shape[:2]
        bg_h, bg_w = bg_img.shape[:2]
        
        if bg_h < h or bg_w < w:
            bg_img = cv2.resize(bg_img, (w, h))
            
        max_y = bg_img.shape[0] - h
        max_x = bg_img.shape[1] - w
        
        start_y = random.randint(0, max_y) if max_y > 0 else 0
        start_x = random.randint(0, max_x) if max_x > 0 else 0
        
        crop = bg_img[start_y:start_y+h, start_x:start_x+w]
        
        # Ekstrakcja tylko ciemnych sygnałów (bez tła) z wycinka
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_crop, 120, 255, cv2.THRESH_BINARY_INV)
        
        alpha = random.uniform(0.3, 0.7)
        result = image.copy()
        
        # Wypalanie tła 
        for c in range(3):
            result[:,:,c] = np.where(
                mask > 0, 
                result[:,:,c] * (1 - alpha) + crop[:,:,c] * alpha, 
                result[:,:,c]
            )
            
        return result.astype(np.uint8)

    def add_vertical_lines(self, image, probability=0.3):
        """
        P_lines = 0.3.
        Dodaje fałszywe, pionowe "zagięcia papieru" i błędy drukarki termicznej,
        aby sieć nie myliła ich z linią izoelektryczną/sygnałem.
        """
        if random.random() > probability:
            return image
            
        result = image.copy()
        h, w = result.shape[:2]
        
        num_lines = random.randint(1, 4)
        for _ in range(num_lines):
            x = random.randint(0, w - 1)
            thickness = random.randint(1, 4)
            intensity = random.randint(30, 180) # Czerń i blada szarość
            
            result[:, x:min(x+thickness, w)] = [intensity, intensity, intensity]
            
        return result
        
    def __call__(self, image):
        """Uruchamia cały reżim augmentacyjny (Adversarial Overlap)"""
        image = self.overlay_signal(image)
        image = self.add_vertical_lines(image)
        return image