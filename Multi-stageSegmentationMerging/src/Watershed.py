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
        self.regions = np.zeros((self.M, self.N))
        self.level = 0
    
    def level_up(self, L):
        self.level += 1
        temp = measure.label(L==self.level)
        for m in range(self.M):
            for n in range(self.N):
                key = temp[m, n]
                if key != 0:
                    flag = False
                    if n-1 >= 0 and self.regions[m, n-1] != 0:
                        self.regions[m, n] = self.regions[m, n-1]
                        flag = True
                    elif n+1 < self.N and self.regions[m, n+1] != 0:
                        self.regions[m, n] = self.regions[m, n+1]
                        flag = True
                    elif m-1 >= 0 and self.regions[m-1, n] != 0:
                        self.regions[m, n] = self.regions[m-1, n]
                        flag = True
                    elif m+1 < self.M and self.regions[m+1, n] != 0:
                        self.regions[m, n] = self.regions[m+1, n]
                        flag = True
                    elif m-1 >= 0 and n-1 >= 0 and self.regions[m-1, n-1] != 0:
                        self.regions[m, n] = self.regions[m-1, n-1]
                        flag = True
                    elif m+1 < self.M and n+1 < self.N and  self.regions[m+1, n+1] != 0:
                        self.regions[m, n] = self.regions[m+1, n+1]
                        flag = True
                    elif m-1 >= 0 and n+1 < self.N and self.regions[m-1, n+1] != 0:
                        self.regions[m, n] = self.regions[m-1, n+1]
                        flag = True
                    elif m+1 < self.M and n-1 >= 0 and  self.regions[m+1, n-1] != 0:
                        self.regions[m, n] = self.regions[m+1, n-1]
                        flag = True
                    
                    if flag == False:
                        self.regions[m, n] = np.max(self.regions)+1

    def get_superpixels(self):
        superpixels = {}
        for m in range(self.M):
            for n in range(self.N):
                if self.regions[m, n] not in superpixels:
                    superpixel = set()
                    superpixel.add((m, n))
                    superpixels[self.regions[m, n]] = superpixel
                    continue
                superpixels[self.regions[m, n]].add((m, n))
        return superpixels
                           
    def run(self):
        sobelgx = ndimage.sobel(self.image, 0)
        sobelgy = ndimage.sobel(self.image, 1)
        g = (sobelgx**2+sobelgy**2)**(1/2)
        plt.figure()
        plt.imshow(g)

        L = np.round(g/self.Q)  # Quantize

        self.regions = measure.label(L==self.level)
        high = np.max(L)
        
        for i in range(int(high)):
            self.level_up(L)
        
        superpixels = self.get_superpixels()      
        return superpixels