import cv2
import numpy as np

# === Load Image ===
img = cv2.imread("sw10005.jpg")  # Replace with your image file
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===  Blur to reduce fine noise ===
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ===  Adaptive Threshold to get yarn (white) vs voids (black) ===
binary = cv2.adaptiveThreshold(
    gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2
)

# ===  Morphological closing to fill small holes ===
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# ===  Remove small black dots inside yarn (white) ===
# Invert to find holes inside yarn
inv = cv2.bitwise_not(closed)

# Find contours of small holes
contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 150:  # Tune this threshold as needed
        cv2.drawContours(inv, [cnt], -1, (0), -1)  # Fill small black specks

# Invert back to get cleaned yarn mask
cleaned = cv2.bitwise_not(inv)

# ===  Calculate Covering Factor ===
yarn_pixels = np.sum(cleaned == 255)
total_pixels = cleaned.size
covering_factor = yarn_pixels / total_pixels * 100

print(f"✅ Cleaned Covering Factor: {covering_factor:.2f}%")
print(f"Yarn Pixels: {yarn_pixels}, Total Pixels: {total_pixels}")

# ===  Show results (optional) ===
def resize(image, scale=0.5):
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)))

cv2.imshow("Original", resize(img))
cv2.imshow("Thresholded", resize(binary))
cv2.imshow("After Closing", resize(closed))
cv2.imshow("Final Cleaned Mask", resize(cleaned))
cv2.waitKey(0)
cv2.destroyAllWindows()


# cv2.imwrite("cleaned_yarn_mask.jpg", cleaned)
