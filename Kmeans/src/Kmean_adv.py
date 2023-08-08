import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
import math
import numpy as np
import random
import time

# mark the points on image 
def image_mark(image, points, label):
    X = []
    Y = []
    for i, j in points:
        X.append(i)
        Y.append(j)

    plt.imshow(image)
    plt.scatter(x=X, y=Y, c="white", label=label)
    plt.title("y")
    plt.legend()
    plt.figure()


def Kmean(image, K, l1, l2, L, iter):
    M, N, O = image.shape
    print(f"shape: {image.shape}")
    W = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
    y = W[0][0]*image[:, :, 0]+W[0][1]*image[:, :, 1]+W[0][2]*image[:, :, 2]
    cb = W[1][0]*image[:, :, 0]+W[1][1]*image[:, :, 1]+W[1][2]*image[:, :, 2]
    cr = W[2][0]*image[:, :, 0]+W[2][1]*image[:, :, 1]+W[2][2]*image[:, :, 2]
    

    # random init points
    def random_K_points(K):
        K_points = []
        while len(K_points) < K:
            point = (random.randint(0, M-1), random.randint(0, N-1))
            if point in K_points:
                continue
            else:
                K_points.append(point)
        return K_points
    
    # Using for adjust init points to min gradient
    def edge_detection(y):
        g_x = ndimage.sobel(y, 0)
        g_y = ndimage.sobel(y, 1)
        g = (g_x**2+g_y**2)**(1/2)
        return g

    def find_min_gradient_point(k, g, L):
        min_g_index = k
        min_g = g[k[0], k[1]]
        for i in range(k[0]-L, k[0]+L+1):
            for j in range(k[1]-L, k[1]+L+1):
                if i < 0 or j < 0 or i > N-1 or j > M-1:
                    continue
                elif g[i, j] < min_g:
                    min_g = g[i, j]
                    min_g_index = (i, j)
        return min_g_index
    
    def set_all_min_gradient_points(K_points, g, L):
        new_K_points = []
        for k in K_points:
            if find_min_gradient_point(k, g, L) in new_K_points or find_min_gradient_point(k, g, L) in K_points:
                new_K_points.append(k)
                continue
            else:
                new_K_points.append(find_min_gradient_point(k, g, L))
        # print(f"new K_points: {new_K_points}")
        return new_K_points
            
    # Calculate each point distance with init points
    def calculate_distance(m, n, regions):
        distances = []       
        for k in regions:
            if m == regions[k]["index"][0][0] and n == regions[k]["index"][0][1]:
                return 
        for k in regions: 
            distances.append((l1*((m-regions[k]["index"][0][0])**2+(n-regions[k]["index"][0][1])**2)+l2*(y[m, n]-regions[k]["y"])**2+(cb[m, n]-regions[k]["cb"])**2+(cr[m, n]-regions[k]["cr"])**2)**(1/2))
        regions[np.argmin(distances)]["index"].append((m, n))

    # After finishing all points distance, calculate new center point from mean value
    def new_center(regions):
        new_m, new_n, new_y, new_cr, new_cb = 0, 0, 0, 0, 0
        for k in regions:
            for m, n in regions[k]["index"]:
                new_m += m
                new_n += n
                new_y += y[m, n]
                new_cb += cb[m, n]
                new_cr += cr[m, n]
            
            new_m = new_m/len(regions[k]["index"])
            new_n = new_n/len(regions[k]["index"])
            new_y = new_y/len(regions[k]["index"])
            new_cb = new_cb/len(regions[k]["index"])
            new_cr = new_cr/len(regions[k]["index"])
            regions[k] = {"y": new_y, "cb": new_cb, "cr": new_cr, "index": [(int(math.floor(new_m)), int(math.floor(new_n)))]}

    # Making the regions map in R matrix
    def regions_map(regions, R):
        for k in regions:
            for m, n in regions[k]["index"]:
                R[m, n] = k

    # Separat the region that do not contact
    def separet_region(R):
        R = measure.label(R)
        regions = {}
        
        for i in range(np.max(R)+1):
            regions[i] = {"index": []}
        for m in range(M):
            for n in range(N):
                regions[R[m, n]]["index"].append((m, n))
        return R, regions
    
    def test_total_points(regions):
        total = 0
        for r in regions:
            total += len(regions[r]["index"])
        print(f"total: {total}")

    def test_find_repeat(regions):
        temp = []
        repeat = []
        for r in regions:
            for i in regions[r]["index"]:
                if i in temp:
                    repeat.append(i)
                temp.append(i)
        print(f"repeat num: {len(repeat)}")
        print(f"repeat: {repeat}")
    
    def test_repeat_points(points):
        for i in range(len(points)-1):
            for j in range(i+1, len(points)):
                if points[i] == points[j]:
                    print(True)
            


    # The main function in kmean function        
    def run():
        R = np.zeros([M, N])
        K_points = random_K_points(K)
        image_mark(image, K_points, "init points")
        g = edge_detection(y)
        K_points = set_all_min_gradient_points(K_points, g, L)
        image_mark(image, K_points, "min gradient points")

        regions = {}
        for k in range(K):
            regions[k] = {"y": y[K_points[k]], "cb": cb[K_points[k]], "cr": cr[K_points[k]], "index": [K_points[k]]}
            
        # print(regions)
        for i in range(iter):   
            test_total_points(regions)
            test_find_repeat(regions)
            for m in range(M):
                for n in range(N):
                    calculate_distance(m, n, regions)
            test_total_points(regions)
            test_find_repeat(regions)
            if i == iter-1:
                break
            else:
                new_center(regions)   
        regions_map(regions, R)
        R, regions = separet_region(R)
        return R, regions
    
    return run()
                
def show_region(region):
    r = []
    for i in region:
        r.append(i)
    return r


def show_segamented_image(image, region, l):
    fig, ax = plt.subplots(4, 4)
    i = 0
    j = 0
    for k in range(16):
        if i // 4 == 1:
            i = 0
            j += 1
        temp = image.copy()
        for m, n in region[l[k]]["index"]:
            temp[m, n] = 255
        ax[i][j].imshow(temp)
        i += 1

def sort_region(region):
    a = []
    for i in region:
        a.append((len(region[i]["index"]), i))
    a = sorted(a, reverse=True)
    sorted_region = {}
    for l, i in a:
        sorted_region[i] = region[i]
    return sorted_region

# read image
path = "Pic/Lena_256bmp.bmp"
image = plt.imread(path)

# main function
start = time.time()
R, regions = Kmean(image, 200, 0.6, 0.8, 10, 10)
end = time.time()
print(f"time: {end- start}")

regions = sort_region(regions)
r = show_region(regions)
show_segamented_image(image, regions, r[0:16])
plt.show()
