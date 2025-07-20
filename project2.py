from skimage import io, color, filters, morphology
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = io.imread("enhanced/sk30003.jpg")

# Convert to grayscale
gray = color.rgb2gray(img)

# Apply Gaussian blur
blurred = filters.gaussian(gray, sigma=12)  # sigma roughly matches kernel size

# Apply Otsu threshold
thresh_val = filters.threshold_triangle(blurred)
thresh = blurred < thresh_val

# Morphological opening to remove noise
opened = morphology.opening(thresh, morphology.square(3))

# Calculate covering factor
yarn_pixels = np.sum(opened)
total_pixels = opened.size
covering_factor = yarn_pixels / total_pixels

print(f"Covering Factor: {covering_factor * 100:.2f}%")
print(f"Yarn Pixels: {yarn_pixels}")
print(f"Total Pixels: {total_pixels}")

# Visualize results
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title("Grayscale")
axes[1].imshow(blurred, cmap='gray')
axes[1].set_title("Blurred")
axes[2].imshow(thresh, cmap='gray')
axes[2].set_title("Thresholded (Inverted)")
axes[3].imshow(opened, cmap='gray')
axes[3].set_title("Morphological Opened")

for ax in axes:
    ax.axis('off')
plt.show()
