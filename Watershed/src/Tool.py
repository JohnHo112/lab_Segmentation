import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def map_regions(regions, M, N):
    R = np.zeros((M, N), dtype=int)
    for r, points in regions.items():
        for point in points:
            R[point[0], point[1]] = r
    return R   

def sort_region(region):
    a = []
    for i in region:
        a.append((len(region[i]), i))
    a = sorted(a, reverse=True)
    sorted_region = {}
    for l, i in a:
        sorted_region[i] = region[i]
    return sorted_region

def show_segamented_image(image, region, l):
    fig, ax = plt.subplots(4, 4)
    i = 0
    j = 0
    for k in range(len(l)):
        if i // 4 == 1:
            i = 0
            j += 1
        temp = image.copy()
        for m, n in region[l[k]]:
            temp[m, n] = 255
        ax[i][j].imshow(temp)
        i += 1

def show_region(region):
    r = []
    for i in region:
        r.append(i)
    return r

def RGB_to_ycbcr(image):
    W = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
    y = W[0][0]*image[:, :, 0]+W[0][1]*image[:, :, 1]+W[0][2]*image[:, :, 2]
    cb = W[1][0]*image[:, :, 0]+W[1][1]*image[:, :, 1]+W[1][2]*image[:, :, 2]
    cr = W[2][0]*image[:, :, 0]+W[2][1]*image[:, :, 1]+W[2][2]*image[:, :, 2]
    return y, cb, cr

def image_gradient(image):
    g_x = ndimage.sobel(image, 0)
    g_y = ndimage.sobel(image, 1)
    g = (g_x**2+g_y**2)**(1/2)
    return g
    