import numpy as np
import matplotlib.pyplot as plt
import cv2

def regions_to_R(regions, M, N):
    R = np.zeros((M, N), dtype=int)
    for r, points in regions.items():
        for point in points:
            R[point[0], point[1]] = r
    return R   

def sort_region(region):
    a = []
    r = []
    for i in region:
        a.append((len(region[i]), i))
    a = sorted(a, reverse=True)
    sorted_region = {}
    for l, i in a:
        sorted_region[i] = region[i]
        r.append(i)
    return sorted_region, r

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

def RGB_to_ycbcr(image):
    W = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
    y = W[0][0]*image[:, :, 0]+W[0][1]*image[:, :, 1]+W[0][2]*image[:, :, 2]
    cb = W[1][0]*image[:, :, 0]+W[1][1]*image[:, :, 1]+W[1][2]*image[:, :, 2]
    cr = W[2][0]*image[:, :, 0]+W[2][1]*image[:, :, 1]+W[2][2]*image[:, :, 2]
    ycbcr = {"y": y, "cb": cb, "cr": cr}
    return ycbcr

def gradient(image, filter):
    return cv2.filter2D(image, -1, filter)

def merge(R, regions, A, B):
        regions[A] = regions[A].union(regions[B])
        for m, n in regions[B]:
            R[m, n] = A
        del regions[B]


def compute_ycbcr_mean(regions, ycbcr):
    meanycbcr = {}
    for r, pixels in regions.items():
        Ay, Acb, Acr = 0, 0, 0
        for m, n in pixels:
            Ay += ycbcr["y"][m, n]
            Acb += ycbcr["cb"][m, n]
            Acr += ycbcr["cr"][m, n]
        Ay = Ay/len(pixels)
        Acb = Acb/len(pixels)
        Acr = Acr/len(pixels)
        meanycbcr[r] = {"y": Ay, "cb": Acb, "cr": Acr}
    return meanycbcr

        
def adjacent_regions(R, regions):
    M, N = R.shape
    adjacent = {}
    for r, pixels in regions.items():
        adj_pixels = set()
        adj_regions = set()
        for m, n in pixels:
            if m-1 >= 0 and R[m-1, n] != R[m, n]:
                adj_pixels.add((m-1, n))
                adj_regions.add(R[m-1, n])
            if m+1 < M and R[m+1, n] != R[m, n]:
                adj_pixels.add((m+1, n)) 
                adj_regions.add(R[m+1, n])
            if n-1 >= 0 and  R[m, n-1] != R[m, n]:
                adj_pixels.add((m, n-1))
                adj_regions.add(R[m, n-1])
            if n+1 < N and R[m, n+1] != R[m, n]:
                adj_pixels.add((m, n+1))
                adj_regions.add(R[m, n+1])
        adjacent[r] = {"adj_pixels": adj_pixels, "adj_regions": adj_regions}
    return adjacent

def find_border(R, regions):
    M, N = R.shape
    regionsBorders = {}
    for r, pixels in regions.items():
        borders = {}
        for m, n in pixels:
            if m-1 >= 0 and R[m-1, n] != R[m, n]:
                if R[m-1, n] in borders:
                    borders[R[m-1, n]].add((m-1, n)) 
                    borders[R[m-1, n]].add((m, n)) 
                else:
                    borders[R[m-1, n]] = set()
                    borders[R[m-1, n]].add((m-1, n))
                    borders[R[m-1, n]].add((m, n)) 

            if m+1 < M and R[m+1, n] != R[m, n]:
                if R[m+1, n] in borders:
                    borders[R[m+1, n]].add((m+1, n)) 
                    borders[R[m+1, n]].add((m, n)) 
                else:
                    borders[R[m+1, n]] = set()
                    borders[R[m+1, n]].add((m+1, n))
                    borders[R[m+1, n]].add((m, n)) 
            if n-1 >= 0 and  R[m, n-1] != R[m, n]:
                if R[m, n-1] in borders:
                    borders[R[m, n-1]].add((m, n-1)) 
                    borders[R[m, n-1]].add((m, n)) 
                else:
                    borders[R[m, n-1]] = set()
                    borders[R[m, n-1]].add((m, n-1))
                    borders[R[m, n-1]].add((m, n)) 
            if n+1 < N and R[m, n+1] != R[m, n]:
                if R[m, n+1] in borders:
                    borders[R[m, n+1]].add((m, n+1)) 
                    borders[R[m, n+1]].add((m, n)) 
                else:
                    borders[R[m, n+1]] = set()
                    borders[R[m, n+1]].add((m, n+1))
                    borders[R[m, n+1]].add((m, n)) 
        regionsBorders[r] = borders
    return regionsBorders

def filter(sigma, L):
    x = np.arange(-L, L+1)
    C = 1/sum(np.exp(1)**(-sigma*x))
    y = C*np.sign(x)*np.exp(1)**(-sigma*x)
    filter = np.array([y, y, y])
    return filter

def process_small_regions_distance(A, B, L):
    return (L["l"]*(A["y"]-B["y"])**2+(A["cb"]-B["cb"])**2+(A["cr"]-B["cr"])**2)**(1/2)

def merge_adjacent_regions_distance(A, B, L, sobelMean, laplaceMean):
    return (L["l1"]*(A["y"]-B["y"])**2+(A["cb"]-B["cb"])**2+(A["cr"]-B["cr"])**2+L["l2"]*sobelMean+L["l3"]*laplaceMean+L["lt"]*((A["tx"]-B["tx"])**2+(A["ty"]-B["ty"])**2))*(0.5)

def border_gradient(border, gradient):
    g = 0
    for m, n in border:
        g += abs(gradient[m, n])
    g = g/len(border)
    return g

def compute_texture(regions, g, r, a):
    texture = 0
    l = len(regions[r])
    for m, n in regions[r]:
        texture += abs(g[m, n])
    texture = (texture/l)**a
    return texture

def convertRegions(regions):
    convertedRegions = {}
    for r, infos in regions.items():
        convertedRegions[r] = set(infos["B"])
    return convertedRegions


def IOU(results, groundTruth, size):
    iou = 0
    for m, Am in groundTruth.items():
        max = 0
        for n, Bn in results.items():
            if len(Am.intersection(Bn)) > max:
                max = len(Am.intersection(Bn))
        iou += max
    iou /= size
    return iou