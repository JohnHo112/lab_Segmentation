import numpy as np
import matplotlib.pyplot as plt
from Watershed import watershed
from MultiStage import process_over_segmentation
import Tool
import time

def main():
    # filter setting
    sobel_x1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/4
    sobel_y1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])/4
    sobel_x2 = np.array([[-1, -1, 0, 1, 1], [-2, -2, 0, 2, 2], [-1, -1, 0, 1, 1]])/8
    sobel_y2 = np.array([[-1, -2, -1], [-1, -2, -1], [0, 0, 0], [1, 2, 1], [1, 2, 1]])/8
    sobel_x3 = np.array([[-1, -1, -1, 0, 1, 1, 1], [-2, -2, -2, 0, 2, 2, 2], [-1, -1, -1, 0, 1, 1, 1]])/12
    sobel_y3 = np.array([[-1, -2, 1], [-1, -2, 1], [-1, -2, 1], [0, 0, 0], [1, 2, 1], [1, 2, 1], [1, 2, 1]])/12
    laplace2 = np.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 24, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])/24
    laplace1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])/8

    L1 = {
        "l": 0.8,
        "l1": 0.6,
        "l2": 0.4,
        "l3": 0.4,
        "lt": 0.4
    }
    L2 = {
        "l": 0.8,
        "l1": 0.5,
        "l2": 0.5,
        "l3": 0.5,
        "lt": 0.5
    }
    L3 = {
        "l": 0.8,
        "l1": 0.4,
        "l2": 0.6,
        "l3": 0.6,
        "lt": 0.6
    }
    L4 = {
        "l": 0.8,
        "l1": 0.3,
        "l2": 0.7,
        "l3": 0.7,
        "lt": 0.7
    }

    # read image
    path = "Pic/Lena_256bmp.bmp"
    # path = "Pic/peppers_s.bmp"

    image = plt.imread(path)
    image = image[128:, :128]
    M, N, O = image.shape

    # convert rgb to ycbcr
    ycbcr = Tool.RGB_to_ycbcr(image)

    # watershed algorithm
    start = time.time()
    regions = watershed(ycbcr["y"], 3).run()
    end = time.time()
    print(f"Watershed time: {end-start}")
    print(f"Watershed regions num: {len(regions)}")

    # convert the results
    R = Tool.regions_to_R(regions, M, N)
    regions, r = Tool.sort_region(regions)

    # show results
    Tool.show_segamented_image(image, regions, r[0:16])

    
    # over segmentation processing
    start = time.time()
    R1, regions1 = process_over_segmentation(R.copy(), regions.copy(), ycbcr, L1, 17, 250, sobel_x3, sobel_y3, laplace2)
    print(f"merge regions num1: {len(regions1)}")
    regions1, r1 = Tool.sort_region(regions1)
    Tool.show_segamented_image(image, regions1, r1[0:16])
    R1, regions1 = process_over_segmentation(R1, regions1, ycbcr, L2, 50, 15, sobel_x2, sobel_y2, laplace2)
    print(f"merge regions num2: {len(regions1)}")
    regions1, r1 = Tool.sort_region(regions1)
    Tool.show_segamented_image(image, regions1, r1[0:16])
    R1, regions1 = process_over_segmentation(R1, regions1, ycbcr, L3, 100, 65, sobel_x1, sobel_y1, laplace1)
    print(f"merge regions num3: {len(regions1)}")
    regions1, r1 = Tool.sort_region(regions1)
    Tool.show_segamented_image(image, regions1, r1[0:16])
    R1, regions1 = process_over_segmentation(R1, regions1, ycbcr, L4, 300, 30, sobel_x1, sobel_y1, laplace1)
    print(f"merge regions num4: {len(regions1)}")
    end = time.time()
    print(f"merge time: {end-start}")

    # convert the results
    R1 = Tool.regions_to_R(regions1, M, N)
    regions1, r1 = Tool.sort_region(regions1)
    plt.figure()
    plt.imshow(R1)

    # show results
    Tool.show_segamented_image(image, regions1, r1[0:16])

    



    

    plt.show()
    
if __name__ == "__main__":
    main()