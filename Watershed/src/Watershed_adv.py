import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import time
from tqdm import tqdm, trange
import Tool

# Watershed 
class Watershed:
    def __init__(self, image):
        self.image = image
        self.M, self.N = image.shape
        self.regions = dict()
    
    def level_up(self, L):
        self.level += 1
        temp = measure.label(L==self.level)
        for m in range(self.M):
            for n in range(self.N):
                key = temp[m, n]
                if key != 0:
                    flag = False
                    for r, points in self.regions.items():
                        for point in points:
                            if (m-1, n) == (point[0], point[1]):
                                self.regions[r].add((m, n))
                                flag = True
                                break
                            elif (m, n-1) == (point[0], point[1]):
                                self.regions[r].add((m, n))
                                flag = True
                                break
                            elif (m+1, n) == (point[0], point[1]):
                                self.regions[r].add((m, n))
                                flag = True
                                break
                            elif (m, n+1) == (point[0], point[1]):
                                self.regions[r].add((m, n))
                                flag = True
                                break                        
                        if flag == True:
                            break
                                
                    if flag == False:
                        pixels = set()
                        pixels.add((m, n))
                        self.regions[max(self.regions)+1] = pixels
                    
    def run(self, Q):
        g = Tool.image_gradient(self.image)  # Compute gradient
        L = np.round(g/Q)  # Quantize
        plt.imshow(L)
        plt.title("L")

        self.level = 0
        temp = measure.label(L==self.level)
        high = np.max(L)
        for m in range(self.M):
            for n in range(self.N):
                key = temp[m, n]
                if key != 0:
                    if key not in self.regions:
                        self.regions[key] = set()
                    self.regions[key].add((m, n))
        
        for _ in tqdm(range(int(high))):
            self.level_up(L)
        
        return self.regions

def distance(Ay, Acb, Acr, By, Bcb, Bcr, l):
    return (l*(Ay-By)**2+(Acb-Bcb)**2+(Acr-Bcr)**2)**(1/2)

def merge(R, regions, A, B):
    regions[A] = regions[A].union(regions[B])
    for m, n in regions[B]:
        R[m, n] = A
    del regions[B]
    return R, regions 

def compute_ycbcr_mean(regison, y, cb, cr):
    meanYCbCr = {}
    for r, pixels in regions.items():
        Ay, Acb, Acr = 0, 0, 0
        for m, n in pixels:
            Ay += y[m, n]
            Acb += cb[m, n]
            Acr += cr[m, n]
        Ay = Ay/len(pixels)
        Acb = Acb/len(pixels)
        Acr = Acr/len(pixels)
        meanYCbCr[r] = {"y": Ay, "cb": Acb, "cr": Acr}
    return meanYCbCr

        
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

def process_small_regions(R, regions, meanYCbCr, delta):
    adjacent = adjacent_regions(R, regions)
    regions_to_merge = []
    for r, pixels in regions.items():
        if len(pixels) < delta:
            Ay, Acb, Acr = meanYCbCr[r]["y"], meanYCbCr[r]["cb"], meanYCbCr[r]["cr"]
            minDiff = 10000000
            minAdj = -1
            for adj in adjacent[r]["adj_regions"]:
                By, Bcb, Bcr = meanYCbCr[adj]["y"], meanYCbCr[adj]["cb"], meanYCbCr[adj]["cr"]
                dist = distance(Ay, Acb, Acr, By, Bcb, Bcr, 1)
                if dist < minDiff:
                    minDiff = dist
                    minAdj = adj
            if minAdj != -1:
                regions_to_merge.append((r, minAdj))

    for r, minAdj in regions_to_merge:
        if r in regions and minAdj in regions:
            if r == minAdj:
                continue
            R, regions = merge(R, regions, r, minAdj)
            for i in range(len(regions_to_merge)):
                if regions_to_merge[i][0] == minAdj:
                    regions_to_merge[i] = (r, regions_to_merge[i][1])
                if regions_to_merge[i][1] == minAdj:
                    regions_to_merge[i] = (regions_to_merge[i][0], r)
    return R, regions
            
def merge_adjacent_regions(R, regions, meanYCbCr, threshold):
    adjacent = adjacent_regions(R, regions)
    regions_to_merge = []
    md = 0
    for r, pixels in regions.items():
        Ay, Acb, Acr = meanYCbCr[r]["y"], meanYCbCr[r]["cb"], meanYCbCr[r]["cr"]
        for adj in adjacent[r]["adj_regions"]:
            By, Bcb, Bcr = meanYCbCr[adj]["y"], meanYCbCr[adj]["cb"], meanYCbCr[adj]["cr"]
            dist = distance(Ay, Acb, Acr, By, Bcb, Bcr, 0.8)
            if dist < threshold:
                regions_to_merge.append((r, adj))

    for r, adj in regions_to_merge:
        if r in regions and adj in regions:
            if r == adj:
                continue
            R, regions = merge(R, regions, r, adj)
            for i in range(len(regions_to_merge)):
                if regions_to_merge[i][0] == adj:
                    regions_to_merge[i] = (r, regions_to_merge[i][1])
                if regions_to_merge[i][1] == adj:
                    regions_to_merge[i] = (regions_to_merge[i][0], r)
    return R, regions

def process_over_segmentation(R, regions, y, cb, cr, delta, threshold):
    meanYCbCr = compute_ycbcr_mean(regions, y, cb, cr)
    for _ in range(10):
        R, regions= process_small_regions(R, regions, meanYCbCr, delta)
    meanYCbCr = compute_ycbcr_mean(regions, y, cb, cr)
    R, regions = merge_adjacent_regions(R, regions, meanYCbCr, threshold)



    
if __name__ == '__main__':
    # read image
    path = "Pic/Lena_256bmp.bmp"
    image = plt.imread(path)
    image = image[:128, :128]
    M, N, O = image.shape

    # rgb image to y image
    y, cb, cr = Tool.RGB_to_ycbcr(image)
    plt.imshow(image)
    plt.title("Image")
    plt.figure()

    # watershed
    start = time.time()
    w = Watershed(y)
    regions = w.run(3)
    print(f"regions num: {len(regions)}")
    end = time.time()
    print(f"Watershed time: {end-start}")

    # convert result
    regions = Tool.sort_region(regions)
    R = Tool.map_regions(regions, M, N)

    r = Tool.show_region(regions)
    Tool.show_segamented_image(y, regions, r[0:16])
    
    print(f"regions num: {len(regions)}")

    # processing watershed over segmentation
    start = time.time()
    process_over_segmentation(R, regions, y, cb, cr, 100, 1)
    end = time.time()
    print(f"process over segmetation time: {end-start}")

    print(f"regions num: {len(regions)}")

    # convert result
    regions = Tool.sort_region(regions)
    R = Tool.map_regions(regions, M, N)

    # show the result plot
    r = Tool.show_region(regions)
    Tool.show_segamented_image(y, regions, r[0:16])
    plt.show()
