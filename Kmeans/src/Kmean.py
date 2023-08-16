import matplotlib.pyplot as plt
import numpy as np
import random
import time

def Kmean(image, K, l1, l2, iter):
    M, N, O = image.shape
    print(f"shape: {image.shape}")
    W = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
    y = W[0][0]*image[:, :, 0]+W[0][1]*image[:, :, 1]+W[0][2]*image[:, :, 2]
    cb = W[1][0]*image[:, :, 0]+W[1][1]*image[:, :, 1]+W[1][2]*image[:, :, 2]
    cr = W[2][0]*image[:, :, 0]+W[2][1]*image[:, :, 1]+W[2][2]*image[:, :, 2]
    R = np.zeros([M, N])

    def random_K_points(K):
        K_points = []
        while len(K_points) < K:
            point = (random.randint(0, M-1), random.randint(0, N-1))
            if point in K_points:
                continue
            else:
                K_points.append(point)
        # print(f"K set: {K_points}")
        return K_points
    
    def calculate_distance(m, n, regions):
        distances = []
        #print(regions)
        for k in regions:
            distances.append((l1*((m-regions[k]["index"][0][0])**2+(n-regions[k]["index"][0][1])**2)+l2*(y[m, n]-regions[k]["y"])**2+(cb[m, n]-regions[k]["cb"])**2+(cr[m, n]-regions[k]["cr"])**2)**(1/2))
        regions[np.argmin(distances)]["index"].append((m, n))

    def new_center():
        for k in regions:
            new_m, new_n, new_y, new_cr, new_cb = 0, 0, 0, 0, 0
            for m, n in regions[k]["index"]:
                new_m += m
                new_n += n
                new_y += y[m][n]
                new_cb += cb[m][n]
                new_cr += cr[m][n]
            
            new_m = new_m/len(regions[k]["index"])
            new_n = new_n/len(regions[k]["index"])
            new_y = new_y/len(regions[k]["index"])
            new_cb = new_cb/len(regions[k]["index"])
            new_cr = new_cr/len(regions[k]["index"])
            regions[k] = {"y": new_y, "cb": new_cb, "cr": new_cr, "index": [(int(new_m), int(new_n))]}


    def regions_map():
        for k in regions:
            for m, n in regions[k]["index"]:
                R[m, n] = k

    # init
    K_points = random_K_points(K)
    regions = {}
    for k in range(K):
        regions[k] = {"y": y[K_points[k]], "cb": cb[K_points[k]], "cr": cr[K_points[k]], "index": [K_points[k]]}
        
    # sub main
    for i in range(iter): 
        print(f"iter: {i}")   
        for m in range(M):
            for n in range(N):
                calculate_distance(m, n, regions)
        if i == iter-1:
            break
        else:
            new_center()

        # print(regions)    
    regions_map()

    return R, regions
                
def show_region(region):
    r = []
    for i in region:
        r.append(i)
    # print(r)
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

# read image
path = "Pic/Lena_256bmp.bmp"
# path = "Pic/peppers.bmp"
image = plt.imread(path)

# main function
start = time.time()
R, regions = Kmean(image, 16, 0.6, 0.8, 10)
r = show_region(regions)
end = time.time()
print(f"time: {end- start}")

show_segamented_image(image, regions, r[0:16])
plt.show()
