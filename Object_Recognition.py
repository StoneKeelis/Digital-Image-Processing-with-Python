import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology, color
from skimage.color import rgb2gray, label2rgb

plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams['image.cmap'] = 'gray'

I = io.imread("makaroni.JPG")

if I.ndim == 3:
    I = rgb2gray(I)

G = filters.gaussian(I, sigma=2)

thresh = filters.threshold_otsu(G)

G_TH = G < thresh

SE = morphology.disk(4)

cleaned = morphology.closing(G_TH, SE)
cleaned = morphology.opening(cleaned, SE)
cleaned = morphology.remove_small_objects(cleaned, min_size=200)
cleaned = morphology.remove_small_holes(cleaned, area_threshold=300)

labels = measure.label(cleaned)
overlay = color.label2rgb(labels, image=I, bg_label=0)

regions = measure.regionprops(labels)
areas = np.array([r.area for r in regions], dtype=float)
areas = areas[areas >= 100]
min_area = max(100, 0.5 * np.median(areas))
regions = sorted([r for r in regions if r.area >= min_area], key=lambda r: (r.centroid[0], r.centroid[1]))


plt.subplot(1,3,1)
plt.imshow(I)
plt.title("Original image")
plt.axis("off");

plt.subplot(1,3,2)
plt.imshow(I)
plt.contour(cleaned, color="red", linewidths=2)
plt.title("Original image with contours")
plt.axis("off");

ax = plt.subplot(1,3,3)
ax.imshow(overlay)
ax.set_title(f"Original image with coloured objects ({len(regions)})")
ax.axis("off");

i = 1
for r in regions:
    (y, x) = r.centroid
    ax.text(x, y, str(i), color="white")
    i += 1

plt.show()
