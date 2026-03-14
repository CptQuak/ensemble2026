import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.binarization import ECGPreprocessor

img_path = "data/small_train/ecg_train_0048.png"
img_bgr = cv2.imread(img_path)
mask = ECGPreprocessor.process(img_bgr, "test_rec", debug_folder="output/debug/temp")

# 1. Raw Sum (Current)
proj_sum = np.sum(mask, axis=1)

# 2. Variance
proj_var = np.var(mask, axis=1)

# 3. Transitions
proj_trans = np.sum(np.abs(np.diff(mask.astype(np.int16), axis=1)), axis=1)

# 4. Vertical Opening (remove horizontal lines)
vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
mask_no_horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk)
proj_no_horiz = np.sum(mask_no_horiz, axis=1)

plt.figure(figsize=(12, 10))

plt.subplot(141)
plt.plot(proj_sum, np.arange(len(proj_sum)))
plt.gca().invert_yaxis()
plt.title("Sum")

plt.subplot(142)
plt.plot(proj_var, np.arange(len(proj_var)))
plt.gca().invert_yaxis()
plt.title("Variance")

plt.subplot(143)
plt.plot(proj_trans, np.arange(len(proj_trans)))
plt.gca().invert_yaxis()
plt.title("Transitions")

plt.subplot(144)
plt.plot(proj_no_horiz, np.arange(len(proj_no_horiz)))
plt.gca().invert_yaxis()
plt.title("No Horiz Sum")

plt.tight_layout()
plt.savefig("test_projections.png")
