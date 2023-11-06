import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

def R_to_regions(R):
  M, N = R.shape
  superpixels = {}
  for m in range(M):
      for n in range(N):
          if R[m, n] not in superpixels:
              superpixel = set()
              superpixel.add((m, n))
              superpixels[R[m, n]] = superpixel
              continue
          superpixels[R[m, n]].add((m, n))
  return superpixels

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

def compute_RGB_mean(regions, RGB):
    # print(RGB.shape)
    meanRGB = {}
    for r, pixels in regions.items():
        AR, AG, AB = 0, 0, 0
        for m, n in pixels:
            AR += RGB[m, n, 0]
            AG += RGB[m, n, 1]
            AB += RGB[m, n, 2]
        AR = AR/len(pixels)
        AG = AG/len(pixels)
        AB = AB/len(pixels)
        meanRGB[r] = {"R": AR, "G": AG, "B": AB}
    return meanRGB

def merge_regions(regions_to_merge, R, regions):
    for r, minAdj in regions_to_merge:
        if r in regions and minAdj in regions:
            if r == minAdj:
                continue
            merge(R, regions, r, minAdj)
            for i in range(len(regions_to_merge)):
                if regions_to_merge[i][0] == minAdj:
                    regions_to_merge[i] = (r, regions_to_merge[i][1])
                if regions_to_merge[i][1] == minAdj:
                    regions_to_merge[i] = (regions_to_merge[i][0], r)



def merge(R, regions, A, B):
        regions[A] = regions[A].union(regions[B])
        for m, n in regions[B]:
            R[m, n] = A
        del regions[B]

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

def gradient(image, filter):
    return cv2.filter2D(image, -1, filter)

def filter(sigma, L):
    x = np.arange(-L, L+1)
    C = 1/sum(np.exp(1)**(-sigma*x))
    y = C*np.sign(x)*np.exp(1)**(-sigma*x)
    filter = np.array([y, y, y])
    return filter

def compute_center(regions):
    centers = {}
    for r, pixels in regions.items():
        center_x = 0
        center_y = 0
        for m, n  in pixels:
            center_x += m
            center_y += n
        centers[r] = (int(center_x/len(pixels)), int(center_y/len(pixels)))
    return centers

def color_distance(A, B):
    return ((A["R"]-B["R"])**2+(A["G"]-B["G"])**2+(A["B"]-B["B"])**2)**(1/2)

def ycbcr_distance(A, B):
    return ((A["y"]-B["y"])**2+(A["cb"]-B["cb"])**2+(A["cr"]-B["cr"])**2)

def process_small_regions_distance(A, B, L):
    return (L["l"]*(A["y"]-B["y"])**2+(A["cb"]-B["cb"])**2+(A["cr"]-B["cr"])**2)**(1/2)

def merge_adjacent_regions_distance(A, B, L, sobelMean, laplaceMean):
    return (L["ld"]*((A["m"]-B["m"])**2+(A["n"]-B["n"])**2)+L["l1"]*(A["y"]-B["y"])**2+(A["cb"]-B["cb"])**2+(A["cr"]-B["cr"])**2+L["l2"]*(sobelMean)+L["l3"]*laplaceMean+L["lt"]*((A["tx"]-B["tx"])**2+(A["ty"]-B["ty"])**2))*(0.5)

def border_gradient(border, gradient, a):
    g = 0
    if a == True:
        for m, n in border:
            g += abs(gradient[m, n])
    else:
        for m, n in border:
            g += gradient[m, n]

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

def min_pixels(regions):
    min = 10000000
    for pixels in regions.values():
        if len(pixels) < min:
            min = len(pixels)
    print(f"min: {min}")

def final_merge(R, regions, g, meanycbcr, threshold1,  threshold2):
    adjacent = adjacent_regions(R, regions)
    regions_to_merge = []
    record1 = []
    record2 = []
    for r in regions:
        for adj in adjacent[r]["adj_regions"]:
            at = compute_texture(regions, g, r, 0.5)
            bt = compute_texture(regions, g, adj, 0.5)
            record1.append(abs(meanycbcr[r]["y"] - meanycbcr[adj]["y"]))
            record2.append(abs(at-bt))
            dist1 = ((meanycbcr[r]["y"] - meanycbcr[adj]["y"])**2+(meanycbcr[r]["cb"] - meanycbcr[adj]["cb"])**2+(meanycbcr[r]["cr"] - meanycbcr[adj]["cr"])**2)**(0.5)
            dist2 = abs(at-bt)
            if  dist1 < threshold1 and dist2 < threshold2:
                regions_to_merge.append((r, adj))
    plt.figure()
    plt.plot(record1)
    plt.figure()
    plt.plot(record2)
    
    merge_regions(regions_to_merge, R, regions)

def IOU(results, groundTruth, size):
    iou = 0
    for A in groundTruth.values():
        ALen = len(A)
        w = ALen/size
        max = 0
        for B in results.values():
            temp = (len(A.intersection(B)))/(len(A.union(B)))
            if temp > max:
                max = temp
        iou += w*max
    return iou

def mean_gradient(image):
    m, n = image.shape
    sobelgx = ndimage.sobel(image, 0)
    sobelgy = ndimage.sobel(image, 1)
    g = (sobelgx**2+sobelgy**2)**(1/2)
    mean_g = np.sum(g)/(m*n)
    return mean_g