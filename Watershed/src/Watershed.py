import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
import numpy as np
import time
from tqdm import tqdm, trange

# Watershed for gray scalar image
class Watershed:
    def __init__(self, image):
        self.image = image
        self.M, self.N = image.shape
        self.regions = dict()

    def image_gradient(self):
        g_x = ndimage.sobel(self.image, 0)
        g_y = ndimage.sobel(self.image, 1)
        g = (g_x**2+g_y**2)**(1/2)
        return g
    
    def quantize(self, g, Q):
        return np.round(g/Q)
    
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
                                self.regions[r].append((m, n))
                                flag = True
                                break
                            elif (m, n-1) == (point[0], point[1]):
                                self.regions[r].append((m, n))
                                flag = True
                                break
                            elif (m+1, n) == (point[0], point[1]):
                                self.regions[r].append((m, n))
                                flag = True
                                break
                            elif (m, n+1) == (point[0], point[1]):
                                self.regions[r].append((m, n))
                                flag = True
                                break                        
                        if flag == True:
                            break
                                
                    if flag == False:
                        self.regions[max(self.regions)+1] = [(m, n)]
                    
                                
                        
    
    def run(self, Q):
        g = self.image_gradient()
        L = self.quantize(g, Q)
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
                        self.regions[key] = []
                    self.regions[key].append((m, n))
        
        for i in tqdm(range(int(high))):
            self.level_up(L)
        
        return self.regions

def map_regions(regions, M, N):
    R = np.zeros((M, N))
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
    for k in range(16):
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

if __name__ == '__main__':
    # read image
    path = "Pic/Lena_256bmp.bmp"
    image = plt.imread(path)
    # rgb image to y image
    W = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
    image = W[0][0]*image[:, :, 0]+W[0][1]*image[:, :, 1]+W[0][2]*image[:, :, 2]
    image = image[:128, 128:]
    plt.imshow(image)
    plt.title("image")
    plt.figure()
    M, N = image.shape

    # watershed
    start = time.time()
    w = Watershed(image)
    regions = w.run(3)
    print(f"regions num: {len(regions)}")
    end = time.time()
    print(f"time: {end-start}")

    # show result
    regions = sort_region(regions)
    R = map_regions(regions, M, N)
    r = show_region(regions)
    show_segamented_image(image, regions, r[0:16])
    plt.show()
