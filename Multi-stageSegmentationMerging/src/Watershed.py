from skimage import measure
from scipy import ndimage
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

# Watershed 
class watershed:
    def __init__(self, image, Q):
        self.image = image
        self.Q = Q
        self.M, self.N = image.shape
        self.regions = {}
    
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
                    
    def run(self):
        sobelgx = ndimage.sobel(self.image, 0)
        sobelgy = ndimage.sobel(self.image, 1)
        g = (sobelgx**2+sobelgy**2)**(1/2)
        L = np.round(g/self.Q)  # Quantize

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