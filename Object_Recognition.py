import matplotlib.pyplot as plt
import numpy as np
from skimage import io, filters, morphology, measure
from skimage.segmentation import find_boundaries
from skimage.color import label2rgb
from skimage.color import rgb2gray

plt.rcParams["image.cmap"] = "gray"
plt.rcParams["figure.figsize"] = [10,5]


image = io.imread("makaroni.JPG")

def gaussian_filter(image):
    image = filters.gaussian(image, sigma=1.2)
    return image


def to_grayscale(image):
    if image.ndim == 3:
        return rgb2gray(image)
    return image


def otsu_and_morph(img):
    threshold = filters.threshold_otsu(img)
    otsu = img < threshold

    SE = morphology.disk(4)

    otsu_clean = morphology.opening(otsu, SE)
    otsu_clean = morphology.closing(otsu_clean, SE)
    otsu_clean = morphology.remove_small_objects(otsu_clean, min_size=200)
    otsu_clean = morphology.remove_small_holes(otsu_clean, area_threshold=300)
    return otsu_clean


def boundaries_and_overlay(mask):
    boundary = find_boundaries(mask, mode="outer")
    return boundary


def automatic_min_area(regions, frac=0.5, min_abs=100):
    areas = np.array([r.area for r in regions], dtype=float)
    areas = areas[areas >= min_abs]

    if areas.size == 0:
        return min_abs
    
    median = np.median(areas)
    area_min = max(min_abs, frac * median)

    return area_min


def image_processing(og_img):
    print("processing started")

    og_gray = to_grayscale(og_img)
    og_gray_gauss = gaussian_filter(og_gray)

    mask = otsu_and_morph(og_gray_gauss)
    boundary = boundaries_and_overlay(mask)

    labels = measure.label(mask, connectivity=2)
    overlay = label2rgb(labels, image=og_gray, bg_label=0)
    
    regions = measure.regionprops(labels)
    AREA_MIN = automatic_min_area(regions)
    regions_sorted = sorted([r for r in regions if r.area >= AREA_MIN], 
                            key=lambda r: (r.centroid[0], r.centroid[1]))
    
    count_reg = len(regions_sorted)


    plt.subplot(1,3,1)
    plt.imshow(og_img)
    plt.title("Original image")
    plt.axis("off")
    
    plt.subplot(1,3,2)
    plt.imshow(mask)
    plt.title("Binary mask used in overlay")
    plt.axis("off");

    ax3 = plt.subplot(1,3,3)

    ax3.imshow(overlay)
    ax3.contour(boundary.astype(float), 
                levels=[0.5], 
                linewidths=1, 
                colors="hotpink")
    ax3.set_title(f"Overlayed image. Object count: {count_reg}")
    ax3.axis("off");
    
    i = 1
    for r in regions_sorted:
        (y, x) = r.centroid
        ax3.text(x, y, str(i), color="white", ha="center", va="center")
        i += 1

    plt.tight_layout()
    print("Almost ready...")
    plt.show()

image_processing(image)
