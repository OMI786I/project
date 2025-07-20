import cv2
import numpy as np
#sw10005,10001
# ====  Load Image ====
img = cv2.imread("sk10003.jpg")

# ====  Convert to Grayscale ====
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ====  Apply Gaussian Blur ====
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ====  Apply Thresholding ====
# Otsu + Inverted Binary so that dark yarn becomes white (255)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ====  Morphological Opening to Remove Noise ====
kernel = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# ====  Calculate Covering Factor ====
# After inversion, yarn is white (255), so we count white pixels
yarn_pixels = np.sum(opened == 255)
total_pixels = opened.size
covering_factor = yarn_pixels / total_pixels

print(f" Covering Factor  {covering_factor * 100:.2f}%")
print(f"Yarn Pixels (white after inversion): {yarn_pixels}")
print(f"Total Pixels: {total_pixels}")

# ====  Visualization (Resize for Display) ====
def resize(image, scale=0.5):
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)))

cv2.imshow("Original Image", resize(img))
cv2.imshow("Grayscale", resize(gray))
cv2.imshow("Blurred", resize(blurred))
cv2.imshow("Thresholded (Inverted)", resize(thresh))
cv2.imshow("Morphological Opened", resize(opened))

cv2.waitKey(0)
cv2.destroyAllWindows()

# ==== Optional: Save Binary Output ====
# cv2.imwrite("binary_mask_yarn.jpg", opened)
